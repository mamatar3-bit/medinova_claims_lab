"""Structure-aware chunking for Markdown policy documents.

Strategy (a common, exam-relevant pattern):
1. Split on Markdown headings so each chunk stays inside one topic.
2. If a section is still longer than ``max_chars``, split it on paragraph / sentence
   boundaries into windows with ``overlap_chars`` of overlap.
3. Prefix every chunk with its document title and section heading, so the chunk
   still makes sense on its own when it is retrieved.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    section: str
    chunk_index: int
    chunk_text: str

    def to_dict(self) -> dict:
        return asdict(self)


def _split_sections(text: str) -> tuple[list[tuple[str, str]], str]:
    """Return ([(heading, body)], title). Text before the first ## heading goes under the title."""
    title = ""
    sections: list[tuple[str, list[str]]] = []
    current_heading, current_lines = "", []
    for line in text.splitlines():
        m = HEADING_RE.match(line.strip())
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = heading
                current_heading = heading
                continue
            if current_lines and any(ln.strip() for ln in current_lines):
                sections.append((current_heading, current_lines))
            current_heading, current_lines = heading, []
        else:
            current_lines.append(line)
    if current_lines and any(ln.strip() for ln in current_lines):
        sections.append((current_heading, current_lines))
    return [(h, "\n".join(lines).strip()) for h, lines in sections], title


def _window(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split long text into overlapping windows, preferring sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    windows, current = [], ""
    for s in sentences:
        if not s:
            continue
        if len(current) + len(s) + 1 > max_chars and current:
            windows.append(current.strip())
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = (tail + " " + s).strip()
        else:
            current = (current + " " + s).strip()
    if current:
        windows.append(current.strip())
    return windows


def chunk_id_for(doc_name: str, index: int, text: str) -> str:
    """Deterministic id: re-running on unchanged text gives the same ids (needed for MERGE)."""
    digest = hashlib.sha1(f"{doc_name}|{index}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{doc_name}-{index:03d}-{digest}"


def chunk_markdown(doc_name: str, text: str, max_chars: int = 1000, overlap_chars: int = 150) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")
    sections, title = _split_sections(text)
    chunks: list[Chunk] = []
    for heading, body in sections:
        for piece in _window(body, max_chars, overlap_chars):
            prefix = f"{title} > {heading}" if title and heading != title else (title or heading)
            chunk_text = f"{prefix}\n\n{piece}" if prefix else piece
            idx = len(chunks)
            chunks.append(Chunk(chunk_id_for(doc_name, idx, chunk_text), doc_name, heading, idx, chunk_text))
    return chunks
