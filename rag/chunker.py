"""Section-aware markdown chunker for the RAG pipeline.

Splits on H1/H2/H3 headers; if a section is still longer than MAX_TOKENS,
falls back to a sliding 800-token window with 150-token overlap.
Each chunk carries enough metadata to cite its source at answer time.
"""
from __future__ import annotations
import os, re, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import tiktoken

MAX_TOKENS = 800
OVERLAP = 150
ENC = tiktoken.get_encoding("cl100k_base")

HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.M)


@dataclass
class Chunk:
    id: str
    path: str       # relative path inside corpus/
    section: str    # nearest header chain, e.g. "Agentic Frameworks > Key claims"
    text: str
    n_tokens: int


def _split_on_headers(md: str) -> list[tuple[str, str]]:
    """Return list of (header_chain, section_text) pairs."""
    positions = [(m.start(), len(m.group(1)), m.group(2).strip()) for m in HEADER_RE.finditer(md)]
    if not positions:
        return [("", md)]
    chunks: list[tuple[str, str]] = []
    stack: list[str] = []
    prev_end = positions[0][0]
    # preamble before first header
    if prev_end > 0:
        chunks.append(("(preamble)", md[:prev_end].strip()))
    for i, (start, level, title) in enumerate(positions):
        # update header stack to `level-1` depth then push
        while len(stack) >= level:
            stack.pop()
        stack.append(title)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(md)
        body = md[start:end].strip()
        chain = " > ".join(stack)
        chunks.append((chain, body))
    return chunks


def _window(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> list[str]:
    toks = ENC.encode(text)
    if len(toks) <= max_tokens:
        return [text]
    out = []
    step = max_tokens - overlap
    for i in range(0, len(toks), step):
        piece = ENC.decode(toks[i : i + max_tokens])
        out.append(piece)
        if i + max_tokens >= len(toks):
            break
    return out


def chunk_file(path: Path, corpus_root: Path) -> Iterable[Chunk]:
    md = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(corpus_root))
    sections = _split_on_headers(md)
    n = 0
    for chain, body in sections:
        for sub in _window(body):
            cid = f"{rel}::{chain}::{n}"
            yield Chunk(id=cid, path=rel, section=chain, text=sub, n_tokens=len(ENC.encode(sub)))
            n += 1


def chunk_corpus(corpus_root: Path) -> list[Chunk]:
    chunks = []
    for md in sorted(corpus_root.rglob("*.md")):
        if md.name.startswith("_"):  # skip _Sidebar, _Footer etc.
            continue
        for c in chunk_file(md, corpus_root):
            if c.text.strip():
                chunks.append(c)
    return chunks


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    chunks = chunk_corpus(root)
    print(f"{len(chunks)} chunks from {len({c.path for c in chunks})} files")
    print(f"avg tokens/chunk: {sum(c.n_tokens for c in chunks) / len(chunks):.0f}")
    print(f"total tokens: {sum(c.n_tokens for c in chunks)}")
