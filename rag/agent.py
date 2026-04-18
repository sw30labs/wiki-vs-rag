"""RAG query-time agent.

Given a query: embed it, top-k FAISS lookup, pack chunks, single Claude call.
Returns answer + every measurement we care about.
"""
from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import faiss
import anthropic
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "claude-sonnet-4-6"
TOP_K = 5
MAX_OUT = 700
# Claude Sonnet 4.6 published prices (per 1M tokens)
SONNET_IN = 3.0
SONNET_OUT = 15.0
EMBED_PRICE = 0.02

SYSTEM = """You are a helpful assistant answering questions about the sw30labs organization wiki.

Below are passages retrieved from the wiki that may be relevant to the user's question.
Answer ONLY using information in these passages. If the passages don't contain the answer, say so plainly.
Cite the source `path` for every factual claim, inline in square brackets like [path/to/file.md].
Be concise — 2-6 sentences unless the question genuinely needs more."""


@dataclass
class RagResult:
    query: str
    answer: str
    retrieved_paths: list[str]
    input_tokens: int
    output_tokens: int
    embed_tokens: int
    cost_usd: float
    latency_s: float
    turns: int = 1


def _load_index(index_dir: Path):
    idx = faiss.read_index(str(index_dir / "index.faiss"))
    chunks = [json.loads(l) for l in (index_dir / "chunks.jsonl").read_text().splitlines()]
    return idx, chunks


def answer(query: str, index_dir: Path, k: int = TOP_K) -> RagResult:
    idx, chunks = _load_index(index_dir)
    oc = OpenAI()
    ac = anthropic.Anthropic()

    t0 = time.perf_counter()
    # embed query
    e = oc.embeddings.create(model=EMBED_MODEL, input=query)
    qvec = np.array([e.data[0].embedding], dtype="float32")
    faiss.normalize_L2(qvec)
    embed_tokens = e.usage.total_tokens

    # retrieve
    scores, ids = idx.search(qvec, k)
    retrieved = [chunks[i] for i in ids[0]]

    # build prompt
    context = "\n\n---\n\n".join(
        f"[source: {c['path']} · section: {c['section']}]\n{c['text']}"
        for c in retrieved
    )
    user = f"Passages:\n\n{context}\n\n---\n\nQuestion: {query}"

    r = ac.messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_OUT,
        system=SYSTEM,
        temperature=0,
        messages=[{"role": "user", "content": user}],
    )
    latency = time.perf_counter() - t0
    answer_text = r.content[0].text

    cost = (
        r.usage.input_tokens / 1_000_000 * SONNET_IN
        + r.usage.output_tokens / 1_000_000 * SONNET_OUT
        + embed_tokens / 1_000_000 * EMBED_PRICE
    )

    return RagResult(
        query=query,
        answer=answer_text,
        retrieved_paths=[c["path"] for c in retrieved],
        input_tokens=r.usage.input_tokens,
        output_tokens=r.usage.output_tokens,
        embed_tokens=embed_tokens,
        cost_usd=round(cost, 6),
        latency_s=round(latency, 2),
    )


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What repos are in the OSCAL & Compliance category?"
    idx = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("rag/index")
    res = answer(q, idx)
    print(json.dumps(res.__dict__, indent=2))
