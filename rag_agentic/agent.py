"""Agentic-RAG query-time agent.

Same retrieval substrate as `rag/agent.py` (FAISS over the same index),
but wrapped in a wiki-style tool-call loop so the model can retrieve
multiple times, refine its query, or fall back to reading a full page
when a chunk is ambiguous.

Purpose: isolate the retrieval axis from the turn-budget axis when
comparing against the single-shot RAG and the wiki-nav agents.
"""
from __future__ import annotations
import os, json, time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import faiss
import anthropic
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-6"
MAX_OUT = 700
MAX_TURNS = 8
TOP_K = 10  # wider recall since the model re-ranks by reading
SONNET_IN = 3.0
SONNET_OUT = 15.0
EMBED_PRICE = 0.02

SYSTEM = """You are a helpful assistant answering questions about the sw30labs organization wiki.

You have two tools to gather evidence:
  - `retrieve(query, k)`: semantic search over the wiki; returns up to k passages with source paths and section headers.
  - `read_full_page(path)`: returns the full markdown of a page when chunks are ambiguous or you need surrounding context.

Strategy:
1. Start with ONE `retrieve` call using the user's question (or a slightly rewritten version).
2. If the returned passages clearly answer the question, stop and answer. Do not burn tool calls.
3. If passages hint at the answer but feel incomplete or cross-branch, either:
   - call `retrieve` again with a refined query (different phrasing or narrower term), or
   - call `read_full_page(path)` to load a full source.
4. Cite source paths inline for every factual claim, in square brackets like [path/to/file.md].
5. Answer concisely (2-6 sentences unless the question needs more). If the wiki doesn't cover the question, say so plainly — don't guess.
"""

TOOLS = [
    {
        "name": "retrieve",
        "description": "Semantic search the wiki index. Returns the top-k most relevant passages with source paths and section headers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query — rephrase the user's question or use a more specific term."},
                "k": {"type": "integer", "description": "Number of passages to return (1-10). Default 5.", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_full_page",
        "description": "Read the full markdown content of a single wiki page. Use when a retrieved chunk is ambiguous and you need surrounding context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from the corpus root, e.g. 'wiki/ai-security/ai-security.md'"}
            },
            "required": ["path"],
        },
    },
]


@dataclass
class AgenticRagResult:
    query: str
    answer: str
    retrieved_queries: list[str] = field(default_factory=list)
    pages_read: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    embed_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    turns: int = 0
    hit_turn_cap: bool = False


def _load_index(index_dir: Path):
    idx = faiss.read_index(str(index_dir / "index.faiss"))
    chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text().splitlines()]
    return idx, chunks


def _retrieve(oc: OpenAI, idx, chunks: list[dict], query: str, k: int) -> tuple[str, int]:
    k = max(1, min(int(k), TOP_K))
    e = oc.embeddings.create(model=EMBED_MODEL, input=query)
    qvec = np.array([e.data[0].embedding], dtype="float32")
    faiss.normalize_L2(qvec)
    _, ids = idx.search(qvec, k)
    retrieved = [chunks[i] for i in ids[0]]
    formatted = "\n\n---\n\n".join(
        f"[source: {c['path']} · section: {c['section']}]\n{c['text']}"
        for c in retrieved
    )
    return formatted, e.usage.total_tokens


def _read_full_page(corpus_root: Path, path: str) -> str:
    p = (corpus_root / path).resolve()
    if corpus_root.resolve() not in p.parents and p != corpus_root.resolve():
        return f"error: path outside corpus ({path})"
    if not p.exists() or not p.is_file():
        return f"error: not found ({path})"
    return p.read_text(encoding="utf-8", errors="ignore")


def answer(query: str, corpus_root: Path, index_dir: Path) -> AgenticRagResult:
    idx, chunks = _load_index(index_dir)
    oc = OpenAI()
    ac = anthropic.Anthropic()

    messages: list[dict] = [{"role": "user", "content": query}]
    res = AgenticRagResult(query=query, answer="")
    t0 = time.perf_counter()

    for turn in range(MAX_TURNS):
        res.turns = turn + 1
        r = ac.messages.create(
            model=CHAT_MODEL,
            max_tokens=MAX_OUT,
            system=SYSTEM,
            temperature=0,
            tools=TOOLS,
            messages=messages,
        )
        res.input_tokens += r.usage.input_tokens
        res.output_tokens += r.usage.output_tokens

        if r.stop_reason == "end_turn":
            for b in r.content:
                if b.type == "text":
                    res.answer += b.text
            break

        if r.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": r.content})
            tool_results = []
            for b in r.content:
                if b.type != "tool_use":
                    continue
                if b.name == "retrieve":
                    q = b.input.get("query", "")
                    k = b.input.get("k", 5)
                    res.retrieved_queries.append(q)
                    out, etok = _retrieve(oc, idx, chunks, q, k)
                    res.embed_tokens += etok
                elif b.name == "read_full_page":
                    path = b.input.get("path", "")
                    res.pages_read.append(path)
                    out = _read_full_page(corpus_root, path)
                else:
                    out = f"error: unknown tool {b.name}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": out,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        for b in r.content:
            if b.type == "text":
                res.answer += b.text
        break
    else:
        res.hit_turn_cap = True
        r = ac.messages.create(
            model=CHAT_MODEL,
            max_tokens=MAX_OUT,
            system=SYSTEM + "\n\nYou have reached the tool-call limit. Answer now with what you have.",
            temperature=0,
            messages=messages + [{"role": "user", "content": "Give your final answer now."}],
        )
        res.input_tokens += r.usage.input_tokens
        res.output_tokens += r.usage.output_tokens
        for b in r.content:
            if b.type == "text":
                res.answer += b.text

    res.latency_s = round(time.perf_counter() - t0, 2)
    res.cost_usd = round(
        res.input_tokens / 1_000_000 * SONNET_IN
        + res.output_tokens / 1_000_000 * SONNET_OUT
        + res.embed_tokens / 1_000_000 * EMBED_PRICE,
        6,
    )
    return res


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What repos are in the OSCAL & Compliance category?"
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus")
    index = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("rag/index")
    res = answer(q, corpus, index)
    print(json.dumps(res.__dict__, indent=2))
