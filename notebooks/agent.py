"""
MediNova claims-support agent.

* Framework : LangGraph ReAct agent (the LLM loops: think -> call a tool -> read result -> answer)
* Interface : MLflow ResponsesAgent, so it can be logged, evaluated, traced and served on Databricks
* Tools     : 1. search_policy_documents  (Vector Search retriever over policy chunks)
              2. get_claim_status         (Unity Catalog SQL function)
              3. get_member_benefits      (Unity Catalog SQL function)
* Guardrails: 1. INPUT  - a deterministic check that refuses requests for PHI before the LLM is called
              2. OUTPUT - a deterministic redaction of PHI patterns (DOB, SSN, phone, email, ICD-10)
                 in every answer, because tool data can carry PHI even when the question did not ask
                 for it (e.g. a denial reason that includes a diagnosis code)

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

# --------------------------------------------------------------------------- UC function client
# Set the client explicitly (removes the "UC function client is not set" message and makes
# tool execution use the Databricks client both in notebooks and on Model Serving).
try:
    from unitycatalog.ai.core.base import set_uc_function_client
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient

    set_uc_function_client(DatabricksFunctionClient())
except Exception as e:  # never block the agent from loading
    print(f"Could not set UC function client: {e}")

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
RECURSION_LIMIT = 12  # max think/tool steps per question; stops runaway tool loops

# --------------------------------------------------------------------------- input guardrail
PHI_REQUEST = re.compile(
    r"\b(date of birth|birth ?date|dob|birthday|when was .{0,40} born|social security|ssn|"
    r"home address|mailing address|phone number|email address|contact (details|info(rmation)?)|"
    r"diagnos(is|es|ed)|icd[- ]?10)\b",
    re.IGNORECASE,
)
REFUSAL = (
    "I can't share personal health information such as dates of birth, contact details or "
    "diagnoses in this channel. I can help with claim status, benefits and policy questions "
    "using a claim ID (CLM-123456) or member ID (M123456)."
)

# --------------------------------------------------------------------------- output guardrail
# Deterministic redaction of PHI patterns in the final answer. Claim IDs (CLM-123456), member IDs
# (M123456), CPT codes and dollar amounts are deliberately NOT matched.
PHI_OUTPUT_PATTERNS = [
    ("DOB", re.compile(
        r"\b(?:date of birth|dob|born(?: on)?)\b[:\s]*"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+ \d{1,2},? \d{4})",
        re.IGNORECASE)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # ICD-10-CM codes in dotted form, e.g. M17.11, E11.9, S83.241A
    ("ICD10", re.compile(r"\b[A-TV-Z]\d[0-9AB]\.[0-9A-Z]{1,4}\b")),
]


def redact_phi(text: str) -> tuple[str, list[str]]:
    """Replace PHI patterns with [REDACTED]; return the new text and which kinds were found."""
    found = []
    for label, pattern in PHI_OUTPUT_PATTERNS:
        if pattern.search(text):
            found.append(label)
            text = pattern.sub("[REDACTED]", text)
    return text, found


def _last_user_text(request: ResponsesAgentRequest) -> str:
    for item in reversed([i.model_dump() for i in request.input]):
        if item.get("role") == "user":
            content = item.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def _text_only(messages):
    """Drop reasoning blocks from reasoning-model replies so output content is plain text.

    Reasoning models return content as a list ([{'type': 'reasoning', ...}, {'type': 'text', ...}]),
    which caused the Pydantic 'Expected str' serializer warning and could leak reasoning text.
    """
    cleaned = []
    for m in messages:
        content = getattr(m, "content", None)
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
                if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
            )
            m = m.model_copy(update={"content": text})
        cleaned.append(m)
    return cleaned


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

    def _redact_event(self, event: ResponsesAgentStreamEvent) -> ResponsesAgentStreamEvent:
        """Apply the output guardrail to answer text in a stream event."""
        data = event.model_dump(exclude_none=True)
        found = []

        # streamed text deltas
        if isinstance(data.get("delta"), str):
            data["delta"], f = redact_phi(data["delta"])
            found += f

        # completed message items
        item = data.get("item")
        if isinstance(item, dict) and item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"], f = redact_phi(part["text"])
                    found += f

        if not found:
            return event
        with mlflow.start_span(name="phi_output_guardrail", span_type="GUARDRAIL") as span:
            span.set_outputs({"redacted": sorted(set(found))})
        return ResponsesAgentStreamEvent(**data)

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # 1. Input guardrail: refuse PHI requests before the LLM is called
        if PHI_REQUEST.search(_last_user_text(request)):
            with mlflow.start_span(name="phi_guardrail", span_type="GUARDRAIL") as span:
                span.set_outputs({"blocked": True})
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=self.create_text_output_item(text=REFUSAL, id=str(uuid4())),
            )
            return

        # 2. Run the agent; 3. output guardrail on everything it says
        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        for _, events in self.agent.stream(
            {"messages": messages},
            {"recursion_limit": RECURSION_LIMIT},
            stream_mode=["updates"],
        ):
            for node_data in events.values():
                if not isinstance(node_data, dict) or not node_data.get("messages"):
                    continue
                for event in output_to_responses_items_stream(_text_only(node_data["messages"])):
                    yield self._redact_event(event)


mlflow.langchain.autolog()
AGENT = ClaimsSupportAgent(graph)
mlflow.models.set_model(AGENT)
