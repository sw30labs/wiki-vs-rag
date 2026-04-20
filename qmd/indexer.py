"""Build a QMD collection over a filtered snapshot of the wiki corpus.

One-time cost, excluded from per-query measurement — same bookkeeping
convention as rag/indexer.py.

Pipeline:
  1. Stage corpus/ → qmd/staging/ (skip _-prefixed files, preserve relative paths)
  2. qmd collection add <staging> --name <collection>   (replace if exists)
  3. qmd embed   (populates local BM25 + vector embeddings)
  4. Measure wall-clock + disk footprint + counts → qmd/index_meta.json

QMD retrieval runs fully local (EmbeddingGemma + Qwen3 reranker). Per-query
embedding cost is $0, but the ~2GB GGUF model download and local compute are
real infra costs and are surfaced separately in the README setup footprint.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

import tiktoken

COLLECTION = "wiki-vs-rag"
ENC = tiktoken.get_encoding("cl100k_base")


def _ensure_qmd() -> None:
    if shutil.which("qmd") is None:
        sys.exit("ERROR: `qmd` not on PATH. Run `make qmd-setup` first "
                 "(needs Bun + `@tobilu/qmd`).")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, text=True)


def _stage_corpus(corpus_dir: Path, staging_dir: Path) -> tuple[int, int]:
    """Copy corpus/**/*.md → staging, skipping _-prefixed files. Returns (n_files, tokens)."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    n_files = 0
    total_tokens = 0
    for md in sorted(corpus_dir.rglob("*.md")):
        if md.name.startswith("_"):
            continue
        rel = md.relative_to(corpus_dir)
        dest = staging_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md, dest)
        n_files += 1
        total_tokens += len(ENC.encode(md.read_text(encoding="utf-8", errors="ignore")))
    return n_files, total_tokens


def _collection_exists(name: str) -> bool:
    r = subprocess.run(["qmd", "collection", "list"], capture_output=True, text=True)
    return name in (r.stdout or "")


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def build_index(corpus_dir: Path, out_dir: Path, collection: str = COLLECTION) -> dict:
    _ensure_qmd()
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "staging"

    t_stage = time.perf_counter()
    n_files, corpus_tokens = _stage_corpus(corpus_dir, staging)
    stage_s = round(time.perf_counter() - t_stage, 2)
    print(f"staged {n_files} files, {corpus_tokens} tokens → {staging}")

    t_add = time.perf_counter()
    if _collection_exists(collection):
        print(f"collection '{collection}' already exists — removing before re-add")
        _run(["qmd", "collection", "remove", collection])
    _run(["qmd", "collection", "add", str(staging.resolve()), "--name", collection])
    add_s = round(time.perf_counter() - t_add, 2)

    t_embed = time.perf_counter()
    _run(["qmd", "embed"])
    embed_s = round(time.perf_counter() - t_embed, 2)

    # Warmup: first `qmd query` triggers reranker + query-expansion GGUF downloads
    # (~1-2 GB). Folding this into setup keeps per-query measurement honest and
    # prevents download progress bars from corrupting stdout JSON on the first real query.
    print("warmup: running one qmd query to preload reranker/expansion models...")
    t_warm = time.perf_counter()
    subprocess.run(
        ["qmd", "query", "warmup", "-c", collection, "-n", "1"],
        check=False, text=True,
    )
    warm_s = round(time.perf_counter() - t_warm, 2)

    cache_home = Path(os.environ.get("QMD_HOME") or Path.home() / ".cache" / "qmd")
    models_bytes = _dir_size_bytes(cache_home / "models") if (cache_home / "models").exists() else 0
    cache_total_bytes = _dir_size_bytes(cache_home)

    summary = {
        "collection": collection,
        "n_files": n_files,
        "corpus_tokens": corpus_tokens,
        "staging_dir": str(staging),
        "stage_wall_seconds": stage_s,
        "collection_add_wall_seconds": add_s,
        "embed_wall_seconds": embed_s,
        "warmup_wall_seconds": warm_s,
        "wall_seconds_total": round(stage_s + add_s + embed_s + warm_s, 2),
        "qmd_cache_dir": str(cache_home),
        "gguf_models_bytes": models_bytes,
        "qmd_cache_total_bytes": cache_total_bytes,
        "embedding_cost_usd": 0.0,
        "note": ("QMD retrieval is fully local (BM25 + EmbeddingGemma + Qwen3 reranker). "
                 "Per-query API cost is Sonnet-only. Model download ~2GB GGUF on first embed."),
    }
    (out_dir / "index_meta.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "qmd")
    build_index(corpus, out)
