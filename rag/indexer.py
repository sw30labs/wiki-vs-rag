"""Build a FAISS index over the wiki corpus.

One-time cost, excluded from the per-query measurement.
Logs embedding tokens and $ spent so the README can report setup cost honestly.
"""
from __future__ import annotations
import os, json, time
from pathlib import Path

import numpy as np
import faiss
from openai import OpenAI

from chunker import chunk_corpus, Chunk

EMBED_MODEL = "text-embedding-3-small"
EMBED_PRICE_PER_M = 0.02  # $/1M input tokens (published)
BATCH = 96


def build_index(corpus_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_corpus(corpus_dir)
    print(f"embedding {len(chunks)} chunks...")

    client = OpenAI()
    vectors: list[list[float]] = []
    tokens_total = 0
    t0 = time.perf_counter()

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=[c.text for c in batch],
        )
        for d in resp.data:
            vectors.append(d.embedding)
        tokens_total += resp.usage.total_tokens

    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps({
                "id": c.id, "path": c.path, "section": c.section,
                "text": c.text, "n_tokens": c.n_tokens,
            }) + "\n")

    summary = {
        "n_chunks": len(chunks),
        "n_files": len({c.path for c in chunks}),
        "corpus_tokens": sum(c.n_tokens for c in chunks),
        "embedding_tokens": tokens_total,
        "embedding_cost_usd": round(tokens_total / 1_000_000 * EMBED_PRICE_PER_M, 6),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "model": EMBED_MODEL,
        "dim": arr.shape[1],
    }
    with open(out_dir / "index_meta.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "rag/index")
    build_index(corpus, out)
