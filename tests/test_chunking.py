import pathlib

import pytest

from medinova.chunking import chunk_markdown

DOCS = pathlib.Path(__file__).resolve().parent.parent / "sample_data" / "policy_docs"

SAMPLE = """# Appeals

## Right to appeal
You can appeal within 180 days.

## Decision timelines
Pre-service appeals are decided within 30 days. Post-service appeals within 60 days.
"""


def test_one_chunk_per_section_with_title_prefix():
    chunks = chunk_markdown("appeals.md", SAMPLE)
    assert [c.section for c in chunks] == ["Right to appeal", "Decision timelines"]
    assert chunks[0].chunk_text.startswith("Appeals > Right to appeal")
    assert "180 days" in chunks[0].chunk_text


def test_chunk_ids_are_deterministic_and_unique():
    a = chunk_markdown("appeals.md", SAMPLE)
    b = chunk_markdown("appeals.md", SAMPLE)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_long_sections_are_windowed_with_overlap():
    long_body = " ".join(f"Sentence number {i} about prior authorization." for i in range(80))
    chunks = chunk_markdown("long.md", f"# Long\n\n## Body\n{long_body}", max_chars=300, overlap_chars=60)
    assert len(chunks) > 1
    assert all(len(c.chunk_text) <= 300 + 60 + len("Long > Body\n\n") for c in chunks)


@pytest.mark.parametrize("max_chars,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_markdown("x.md", SAMPLE, max_chars=max_chars, overlap_chars=overlap)


def test_every_sample_policy_doc_chunks_cleanly():
    files = sorted(DOCS.glob("*.md"))
    assert len(files) == 8
    for f in files:
        chunks = chunk_markdown(f.name, f.read_text(encoding="utf-8"))
        assert chunks, f"no chunks for {f.name}"
        assert all(c.chunk_text.strip() for c in chunks)
