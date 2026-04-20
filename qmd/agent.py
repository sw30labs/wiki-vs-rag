"""QMD query-time agent.

Local hybrid retrieval (BM25 FTS5 + EmbeddingGemma + Qwen3 reranker) via a
`qmd query --json` subprocess → single Claude call with retrieved chunks in
context. Structurally parallel to rag/agent.py; only the retrieval step differs.

CHAT_MODEL, MAX_OUT, TOP_K, SONNET_IN/OUT, and SYSTEM must stay verbatim
in sync with rag/agent.py so the synthesis call is identical across arms.
See CLAUDE.md ("Both agents share the same model and temperature").
"""
from __future__ import annotations
import json, shutil, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path

import anthropic

CHAT_MODEL = "claude-sonnet-4-6"
TOP_K = 5
MAX_OUT = 700
SONNET_IN = 3.0
SONNET_OUT = 15.0
EMBED_PRICE = 0.0  # local embeddings; per-query API cost is Sonnet-only
COLLECTION = "wiki-vs-rag"

# VERBATIM from rag/agent.py — do not drift.
SYSTEM = """You are a helpful assistant answering questions about the sw30labs organization wiki.

Below are passages retrieved from the wiki that may be relevant to the user's question.
Answer ONLY using information in these passages. If the passages don't contain the answer, say so plainly.
Cite the source `path` for every factual claim, inline in square brackets like [path/to/file.md].
Be concise — 2-6 sentences unless the question genuinely needs more."""


@dataclass
class QmdResult:
    query: str
    answer: str
    retrieved_paths: list[str]
    input_tokens: int
    output_tokens: int
    embed_tokens: int
    cost_usd: float
    latency_s: float
    retrieval_latency_s: float
    turns: int = 1


def _qmd_query(query: str, collection: str, k: int) -> tuple[list[dict], float]:
    if shutil.which("qmd") is None:
        raise RuntimeError("`qmd` not on PATH — run `make qmd-setup` and `make qmd-index` first.")
    t0 = time.perf_counter()
    r = subprocess.run(
        ["qmd", "query", query, "-c", collection, "-n", str(k), "--json"],
        check=True, capture_output=True, text=True,
    )
    dt = round(time.perf_counter() - t0, 3)
    out = (r.stdout or "").strip()
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"qmd query returned non-JSON: {out[:500]}") from e
    results = data["results"] if isinstance(data, dict) and "results" in data else data
    if not isinstance(results, list):
        raise RuntimeError(f"unexpected qmd --json shape: {type(results).__name__}")
    return results, dt


def _corpus_relative(raw_path: str, staging_root: Path, collection: str) -> str:
    """Normalize QMD's path to a corpus-relative form the judge + report pipeline expects."""
    if not raw_path:
        return ""
    # QMD URI form: qmd://<collection>/<relative-path>   (also tolerate single-slash)
    if raw_path.startswith("qmd:"):
        rest = raw_path[len("qmd:"):].lstrip("/")
        parts = rest.split("/", 1)
        if len(parts) == 2 and parts[0] == collection:
            return parts[1]
        return rest
    p = Path(raw_path)
    # Absolute filesystem path inside staging_root.
    try:
        return str(p.resolve().relative_to(staging_root.resolve()))
    except ValueError:
        pass
    # Relative path with a leading "staging/" segment.
    parts = p.parts
    if parts and parts[0] == "staging":
        return str(Path(*parts[1:]))
    return str(p)


def answer(query: str, corpus_root: Path | None = None, collection: str = COLLECTION, k: int = TOP_K) -> QmdResult:
    ac = anthropic.Anthropic()
    staging_root = Path("qmd") / "staging"

    t_total = time.perf_counter()
    results, retrieval_dt = _qmd_query(query, collection, k)

    formatted, retrieved_paths = [], []
    for r in results:
        path_raw = r.get("displayPath") or r.get("path") or r.get("file") or ""
        path = _corpus_relative(path_raw, staging_root, collection)
        section = r.get("title") or r.get("section") or ""
        body = r.get("snippet") or r.get("context") or r.get("body") or r.get("text") or ""
        if not body:
            continue
        formatted.append(f"[source: {path} · section: {section}]\n{body}")
        retrieved_paths.append(path)

    context = "\n\n---\n\n".join(formatted)
    user = f"Passages:\n\n{context}\n\n---\n\nQuestion: {query}"

    r = ac.messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_OUT,
        system=SYSTEM,
        temperature=0,
        messages=[{"role": "user", "content": user}],
    )
    latency = round(time.perf_counter() - t_total, 2)

    cost = (
        r.usage.input_tokens / 1_000_000 * SONNET_IN
        + r.usage.output_tokens / 1_000_000 * SONNET_OUT
    )

    return QmdResult(
        query=query,
        answer=r.content[0].text,
        retrieved_paths=retrieved_paths,
        input_tokens=r.usage.input_tokens,
        output_tokens=r.usage.output_tokens,
        embed_tokens=0,
        cost_usd=round(cost, 6),
        latency_s=latency,
        retrieval_latency_s=retrieval_dt,
    )


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What repos are in the OSCAL & Compliance category?"
    coll = sys.argv[2] if len(sys.argv) > 2 else COLLECTION
    res = answer(q, collection=coll)
    print(json.dumps(res.__dict__, indent=2))
