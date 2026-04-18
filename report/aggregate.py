"""Aggregate runs + judgments into CSV + PNG charts + headline JSON."""
from __future__ import annotations
import os, json, csv
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def aggregate(runs_path: Path, judgments_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load(runs_path)
    judgments = load(judgments_path)

    # build joined records: (id, agent) -> tokens + quality
    by_key: dict[tuple[str, str], dict] = {}
    for r in runs:
        by_key[(r["id"], r["agent"])] = dict(r)

    # de-blind judgments: map agent -> scores
    for j in judgments:
        for letter in ("A", "B"):
            agent = j[f"{letter.lower()}_agent"]
            scores = j[letter]
            key = (j["id"], agent)
            if key in by_key:
                by_key[key]["relevance"] = scores.get("relevance")
                by_key[key]["completeness"] = scores.get("completeness")
                by_key[key]["groundedness"] = scores.get("groundedness")
                avg = sum(
                    s for s in (scores.get("relevance"), scores.get("completeness"), scores.get("groundedness"))
                    if s is not None
                ) / 3
                by_key[key]["quality_avg"] = round(avg, 2)

    # write master CSV
    fields = [
        "id", "category", "agent", "query",
        "input_tokens", "output_tokens", "total_tokens",
        "turns", "cost_usd", "latency_s",
        "relevance", "completeness", "groundedness", "quality_avg",
    ]
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(by_key.values(), key=lambda r: (r["id"], r["agent"])):
            w.writerow(rec)

    # aggregate stats
    def stats(agent: str, category: str | None = None) -> dict:
        items = [r for (qid, ag), r in by_key.items() if ag == agent
                 and (category is None or r["category"] == category)]
        if not items:
            return {}
        q_items = [r["quality_avg"] for r in items if r.get("quality_avg") is not None]
        return {
            "n": len(items),
            "avg_total_tokens": round(st.mean(r["total_tokens"] for r in items)),
            "median_total_tokens": round(st.median(r["total_tokens"] for r in items)),
            "avg_input_tokens": round(st.mean(r["input_tokens"] for r in items)),
            "avg_output_tokens": round(st.mean(r["output_tokens"] for r in items)),
            "avg_turns": round(st.mean(r["turns"] for r in items), 2),
            "avg_cost_usd": round(st.mean(r["cost_usd"] for r in items), 5),
            "total_cost_usd": round(sum(r["cost_usd"] for r in items), 4),
            "avg_latency_s": round(st.mean(r["latency_s"] for r in items), 2),
            "avg_quality": round(st.mean(q_items), 2) if q_items else None,
            "n_quality_scored": len(q_items),
        }

    summary = {
        "overall": {
            "rag": stats("rag"),
            "wiki": stats("wiki"),
        },
        "by_category": {},
    }
    for cat in ["narrow", "branch", "cross-branch"]:
        summary["by_category"][cat] = {
            "rag": stats("rag", cat),
            "wiki": stats("wiki", cat),
        }

    # head-to-head quality on same query
    wins = {"rag": 0, "wiki": 0, "tie": 0}
    quality_diffs = []
    for qid in sorted({r["id"] for r in runs}):
        rag = by_key.get((qid, "rag"))
        wik = by_key.get((qid, "wiki"))
        if not (rag and wik and rag.get("quality_avg") is not None and wik.get("quality_avg") is not None):
            continue
        dq = wik["quality_avg"] - rag["quality_avg"]
        quality_diffs.append(dq)
        if abs(dq) < 0.17:  # treat diffs <0.17 (0.5 on any one axis) as ties
            wins["tie"] += 1
        elif dq > 0:
            wins["wiki"] += 1
        else:
            wins["rag"] += 1
    summary["head_to_head_quality"] = wins
    summary["quality_diff_mean_wiki_minus_rag"] = round(st.mean(quality_diffs), 2) if quality_diffs else None

    # token ratio
    r_avg = summary["overall"]["rag"]["avg_total_tokens"]
    w_avg = summary["overall"]["wiki"]["avg_total_tokens"]
    summary["wiki_vs_rag_token_ratio"] = round(w_avg / r_avg, 1)
    summary["wiki_vs_rag_cost_ratio"] = round(
        summary["overall"]["wiki"]["avg_cost_usd"] / summary["overall"]["rag"]["avg_cost_usd"], 1
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- charts ----------
    cats = ["narrow", "branch", "cross-branch"]
    # 1. token comparison bar chart by category
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(cats))
    rag_vals = [summary["by_category"][c]["rag"]["avg_total_tokens"] for c in cats]
    wiki_vals = [summary["by_category"][c]["wiki"]["avg_total_tokens"] for c in cats]
    width = 0.38
    b1 = ax.bar([i - width/2 for i in x], rag_vals, width, label="RAG", color="#d97706")
    b2 = ax.bar([i + width/2 for i in x], wiki_vals, width, label="Wiki navigation", color="#2563eb")
    ax.set_xticks(list(x)); ax.set_xticklabels(cats)
    ax.set_ylabel("avg total tokens per query")
    ax.set_title("Tokens per query by category (lower = better)")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "tokens_by_category.png", dpi=140)
    plt.close()

    # 2. quality bar chart by category
    fig, ax = plt.subplots(figsize=(8, 5))
    rag_q = [summary["by_category"][c]["rag"]["avg_quality"] or 0 for c in cats]
    wiki_q = [summary["by_category"][c]["wiki"]["avg_quality"] or 0 for c in cats]
    b1 = ax.bar([i - width/2 for i in x], rag_q, width, label="RAG", color="#d97706")
    b2 = ax.bar([i + width/2 for i in x], wiki_q, width, label="Wiki navigation", color="#2563eb")
    ax.set_xticks(list(x)); ax.set_xticklabels(cats)
    ax.set_ylabel("avg judge score (1-5)")
    ax.set_ylim(0, 5.2)
    ax.set_title("Answer quality by category (higher = better)")
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "quality_by_category.png", dpi=140)
    plt.close()

    # 3. tokens vs quality scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for agent, color in [("rag", "#d97706"), ("wiki", "#2563eb")]:
        pts = [(r["total_tokens"], r.get("quality_avg"))
               for (qid, ag), r in by_key.items() if ag == agent and r.get("quality_avg") is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   c=color, label=agent.upper(), alpha=0.75, s=60, edgecolors="white")
    ax.set_xlabel("total tokens per query")
    ax.set_ylabel("judge quality score (1-5)")
    ax.set_title("Tokens vs. quality — each point is one query")
    ax.set_xscale("log")
    ax.set_ylim(0, 5.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tokens_vs_quality.png", dpi=140)
    plt.close()

    return summary


if __name__ == "__main__":
    import sys
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/runs.jsonl")
    judg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/judgments.jsonl")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("report")
    s = aggregate(runs, judg, out)
    print(json.dumps(s, indent=2))
