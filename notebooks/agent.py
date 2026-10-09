"""
MediNova claims-support agent.

* Framework : LangGraph ReAct agent (the LLM loops: think -> call a tool -> read result -> answer)
* Interface : MLflow ResponsesAgent, so it can be logged, evaluated, traced and served on Databricks
* Tools     : 1. search_policy_documents  (Vector Search retriever over policy chunks)
              2. get_claim_status         (Unity Catalog SQL function)
              3. get_member_benefits      (Unity Catalog SQL function)
* Guardrail : a deterministic input check that refuses requests for PHI before the LLM is called

Settings come from agent_config.yaml (written by notebook 08), so the same code runs in dev,
staging and prod with different catalogs and endpoints.
"""
import re
from typing import Generator
from uuid import uuid4

import mlflow
from databricks_langchain import ChatDatabricks, UCFunctionToolkit, VectorSearchRetrieverTool
from langgraph.prebuilt import create_react_agent
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

config = mlflow.models.ModelConfig(development_config="agent_config.yaml")

# --------------------------------------------------------------------------- tools
retriever = VectorSearchRetrieverTool(
    index_name=config.get("vs_index"),
    tool_name="search_policy_documents",
    tool_description=(
        "Searches MediNova Health policy documents: benefits and copays, prior authorization, "
        "claims submission and timely filing, appeals, network and referrals, denial reason codes, "
        "coverage exclusions and member-services FAQ. Use for any question about rules or policy."
    ),
    num_results=4,
    columns=["doc_name", "section", "chunk_text"],
)
uc_tools = UCFunctionToolkit(function_names=config.get("uc_functions")).tools
TOOLS = [retriever, *uc_tools]

# --------------------------------------------------------------------------- model + graph
llm = ChatDatabricks(endpoint=config.get("llm_endpoint"), temperature=0.0)
graph = create_react_agent(llm, tools=TOOLS, prompt=config.get("system_prompt"))

# --------------------------------------------------------------------------- guardrail
PHI_REQUEST = re.compile(
    r"\b(date of birth|dob|birthday|social security|ssn|home address|phone number|email address|"
    r"diagnos(is|es|ed)|icd[- ]?10)\b",
    re.IGNORECASE,
)
REFUSAL = (
    "I can't share personal health information such as dates of birth, contact details or "
    "diagnoses in this channel. I can help with claim status, benefits and policy questions "
    "using a claim ID (CLM-123456) or member ID (M123456)."
)


def _last_user_text(request: ResponsesAgentRequest) -> str:
    for item in reversed([i.model_dump() for i in request.input]):
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


# --------------------------------------------------------------------------- ResponsesAgent wrapper
class ClaimsSupportAgent(ResponsesAgent):
    def __init__(self, agent):
        self.agent = agent

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        if PHI_REQUEST.search(_last_user_text(request)):
            with mlflow.start_span(name="phi_guardrail", span_type="GUARDRAIL") as span:
                span.set_outputs({"blocked": True})
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=REFUSAL, id=str(uuid4())),
            )
            return

        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        for _, events in self.agent.stream({"messages": messages}, stream_mode=["updates"]):
            for node_data in events.values():
                yield from output_to_responses_items_stream(node_data["messages"])


mlflow.langchain.autolog()
AGENT = ClaimsSupportAgent(graph)
mlflow.models.set_model(AGENT)
