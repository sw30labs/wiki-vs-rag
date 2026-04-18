"""Run both pipelines over the eval query set.

Writes one jsonl line per (query, agent) run.
"""
from __future__ import annotations
import os, json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.agent import answer as rag_answer
from wiki.agent import answer as wiki_answer

BUDGET_USD = 10.0


def run_all(queries_path: Path, corpus_root: Path, index_dir: Path, sitemap_md: Path, out_path: Path) -> dict:
    queries = [json.loads(l) for l in queries_path.read_text().splitlines()]
    sitemap = sitemap_md.read_text()
    spent = 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w")

    stats = {"rag": [], "wiki": []}

    for q in queries:
        qid = q["id"]
        print(f"[{qid}][{q['category']}] {q['query'][:80]}")

        # RAG
        r = rag_answer(q["query"], index_dir)
        spent += r.cost_usd
        rec = {
            "id": qid,
            "category": q["category"],
            "query": q["query"],
            "ground_truth": q["sources"],
            "agent": "rag",
            "answer": r.answer,
            "retrieved_paths": r.retrieved_paths,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "embed_tokens": r.embed_tokens,
            "total_tokens": r.input_tokens + r.output_tokens,
            "turns": r.turns,
            "cost_usd": r.cost_usd,
            "latency_s": r.latency_s,
        }
        f.write(json.dumps(rec) + "\n")
        f.flush()
        stats["rag"].append(rec)
        print(f"  RAG  tok={rec['total_tokens']:5d}  turns={r.turns}  ${r.cost_usd:.4f}  {r.latency_s}s")

        # WIKI
        w = wiki_answer(q["query"], corpus_root, sitemap)
        spent += w.cost_usd
        rec = {
            "id": qid,
            "category": q["category"],
            "query": q["query"],
            "ground_truth": q["sources"],
            "agent": "wiki",
            "answer": w.answer,
            "pages_read": w.pages_read,
            "input_tokens": w.input_tokens,
            "output_tokens": w.output_tokens,
            "total_tokens": w.input_tokens + w.output_tokens,
            "turns": w.turns,
            "cost_usd": w.cost_usd,
            "latency_s": w.latency_s,
            "hit_turn_cap": w.hit_turn_cap,
        }
        f.write(json.dumps(rec) + "\n")
        f.flush()
        stats["wiki"].append(rec)
        print(f"  WIKI tok={rec['total_tokens']:5d}  turns={w.turns}  pages={len(w.pages_read)}  ${w.cost_usd:.4f}  {w.latency_s}s")

        if spent > BUDGET_USD:
            print(f"!!! BUDGET EXCEEDED (${spent:.2f}), aborting")
            break

    f.close()
    summary = {
        "n": len(queries),
        "rag_total_cost": round(sum(r["cost_usd"] for r in stats["rag"]), 4),
        "wiki_total_cost": round(sum(r["cost_usd"] for r in stats["wiki"]), 4),
        "rag_avg_tokens": round(sum(r["total_tokens"] for r in stats["rag"]) / max(1, len(stats["rag"]))),
        "wiki_avg_tokens": round(sum(r["total_tokens"] for r in stats["wiki"]) / max(1, len(stats["wiki"]))),
    }
    (out_path.parent / "run_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    queries = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/queries.jsonl")
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus")
    index_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("rag/index")
    sitemap = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("wiki/sitemap/sitemap.md")
    out = Path(sys.argv[5]) if len(sys.argv) > 5 else Path("runs/runs.jsonl")
    s = run_all(queries, corpus, index_dir, sitemap, out)
    print(json.dumps(s, indent=2))
