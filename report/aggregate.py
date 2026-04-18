"""Aggregate runs + judgments into CSV + PNG charts + headline JSON."""
from __future__ import annotations
import os, json, csv, math
from itertools import combinations
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS = ["rag", "rag_agentic", "wiki"]
AGENT_LABELS = {"rag": "RAG", "rag_agentic": "Agentic-RAG", "wiki": "Wiki nav"}
AGENT_COLORS = {"rag": "#d97706", "rag_agentic": "#059669", "wiki": "#2563eb"}
TIE_THRESHOLD = 0.17  # ≈0.5 on any one axis; matches original setting

# Qualitative scores for axes the benchmark does NOT measure. 0-1, higher=better.
# wiki is the natural winner on all three (infra-free, human-readable, git-push updates).
# rag and rag_agentic share the same FAISS/embedding substrate, so they score the same.
QUALITATIVE_SCORES = {
    "operational_simplicity": {"rag": 0.25, "rag_agentic": 0.25, "wiki": 1.0},
    "human_readable_surface": {"rag": 0.10, "rag_agentic": 0.10, "wiki": 1.0},
    "update_velocity":        {"rag": 0.20, "rag_agentic": 0.20, "wiki": 1.0},
}
QUALITATIVE_AXIS_LABELS = {
    "operational_simplicity": "Operational simplicity",
    "human_readable_surface": "Human-readable surface",
    "update_velocity":        "Update velocity",
}


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _tradeoff_radar_data(summary: dict) -> dict:
    """Build the 6-axis radar dataset: 3 measured (min-max normalized across agents) + 3 scored."""
    overall = summary["overall"]

    def minmax_higher_better(vals: dict[str, float]) -> dict[str, float]:
        lo, hi = min(vals.values()), max(vals.values())
        return {k: (v - lo) / (hi - lo) if hi > lo else 0.5 for k, v in vals.items()}

    def minmax_lower_better(vals: dict[str, float]) -> dict[str, float]:
        lo, hi = min(vals.values()), max(vals.values())
        return {k: 1 - (v - lo) / (hi - lo) if hi > lo else 0.5 for k, v in vals.items()}

    q = minmax_higher_better({a: overall[a]["avg_quality"] for a in AGENTS})
    c = minmax_lower_better({a: overall[a]["avg_cost_usd"] for a in AGENTS})
    l = minmax_lower_better({a: overall[a]["avg_latency_s"] for a in AGENTS})

    return {
        agent: {
            "Quality":                q[agent],
            "Cost efficiency":        c[agent],
            "Latency":                l[agent],
            "Update velocity":        QUALITATIVE_SCORES["update_velocity"][agent],
            "Human-readable surface": QUALITATIVE_SCORES["human_readable_surface"][agent],
            "Operational simplicity": QUALITATIVE_SCORES["operational_simplicity"][agent],
        }
        for agent in AGENTS
    }


def _draw_radar(radar_data: dict, overall: dict, out_path: Path) -> None:
    # Clockwise from 12 o'clock: measured axes on the top/right, qualitative on the bottom/left.
    axes_order = [
        "Quality", "Cost efficiency", "Latency",
        "Update velocity", "Human-readable surface", "Operational simplicity",
    ]
    n = len(axes_order)
    angles = [i * 2 * math.pi / n for i in range(n)]
    angles_closed = angles + [angles[0]]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise

    for agent in AGENTS:
        vals = [radar_data[agent][a] for a in axes_order]
        vals_closed = vals + [vals[0]]
        ax.plot(angles_closed, vals_closed, color=AGENT_COLORS[agent], linewidth=2, label=AGENT_LABELS[agent])
        ax.fill(angles_closed, vals_closed, color=AGENT_COLORS[agent], alpha=0.18)

    ax.set_xticks(angles)
    ax.set_xticklabels(axes_order, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.10), fontsize=10)
    ax.set_title(
        "Tradeoff shape — outer = better\n"
        "Measured axes (top/right) min-max normalized across the three agents.\n"
        "Qualitative axes (bottom/left) are author-scored 0–1.",
        fontsize=10, y=1.10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


def _draw_qualitative_heatmap(out_path: Path) -> None:
    axes = list(QUALITATIVE_SCORES.keys())
    matrix = [[QUALITATIVE_SCORES[axis][agent] for agent in AGENTS] for axis in axes]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(AGENTS)))
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENTS])
    ax.set_yticks(range(len(axes)))
    ax.set_yticklabels([QUALITATIVE_AXIS_LABELS[a] for a in axes])

    for i, axis in enumerate(axes):
        for j, agent in enumerate(AGENTS):
            v = matrix[i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else "black", fontsize=11)

    plt.colorbar(im, ax=ax, label="score (0–1, higher = better)")
    ax.set_title("Qualitative axes (author-scored, not measured)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def aggregate(runs_path: Path, judgments_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load(runs_path)
    judgments = load(judgments_path)

    # build joined records: (id, agent) -> tokens + quality
    by_key: dict[tuple[str, str], dict] = {}
    for r in runs:
        by_key[(r["id"], r["agent"])] = dict(r)

    # absolute-scoring judgments: direct (id, agent) lookup
    for j in judgments:
        key = (j["id"], j["agent"])
        if key not in by_key:
            continue
        scores = j.get("scores", {})
        by_key[key]["relevance"] = scores.get("relevance")
        by_key[key]["completeness"] = scores.get("completeness")
        by_key[key]["groundedness"] = scores.get("groundedness")
        present = [s for s in (scores.get("relevance"), scores.get("completeness"), scores.get("groundedness")) if s is not None]
        if present:
            by_key[key]["quality_avg"] = round(sum(present) / len(present), 2)

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
        "overall": {agent: stats(agent) for agent in AGENTS},
        "by_category": {},
    }
    for cat in ["narrow", "branch", "cross-branch"]:
        summary["by_category"][cat] = {agent: stats(agent, cat) for agent in AGENTS}

    # pairwise head-to-head quality on same query
    summary["head_to_head"] = {}
    for a, b in combinations(AGENTS, 2):
        a_wins, b_wins, ties = 0, 0, 0
        diffs = []
        for qid in sorted({r["id"] for r in runs}):
            ra = by_key.get((qid, a))
            rb = by_key.get((qid, b))
            if not (ra and rb and ra.get("quality_avg") is not None and rb.get("quality_avg") is not None):
                continue
            dq = rb["quality_avg"] - ra["quality_avg"]
            diffs.append(dq)
            if abs(dq) < TIE_THRESHOLD:
                ties += 1
            elif dq > 0:
                b_wins += 1
            else:
                a_wins += 1
        summary["head_to_head"][f"{a}_vs_{b}"] = {
            "a": a, "b": b,
            "a_wins": a_wins, "b_wins": b_wins, "ties": ties,
            "mean_quality_diff_b_minus_a": round(st.mean(diffs), 2) if diffs else None,
        }

    # pairwise token / cost ratios (b / a for every ordered pair)
    summary["pairwise_ratios"] = {}
    for a in AGENTS:
        for b in AGENTS:
            if a == b:
                continue
            a_tok = summary["overall"][a].get("avg_total_tokens") or 0
            b_tok = summary["overall"][b].get("avg_total_tokens") or 0
            a_cost = summary["overall"][a].get("avg_cost_usd") or 0
            b_cost = summary["overall"][b].get("avg_cost_usd") or 0
            summary["pairwise_ratios"][f"{b}_over_{a}"] = {
                "token_ratio": round(b_tok / a_tok, 2) if a_tok else None,
                "cost_ratio": round(b_cost / a_cost, 2) if a_cost else None,
            }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------- charts ----------
    cats = ["narrow", "branch", "cross-branch"]
    n_agents = len(AGENTS)
    total_width = 0.8
    width = total_width / n_agents
    x = list(range(len(cats)))

    def offset(i: int) -> float:
        return (i - (n_agents - 1) / 2) * width

    # 1. tokens by category
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, agent in enumerate(AGENTS):
        vals = [summary["by_category"][c][agent].get("avg_total_tokens", 0) for c in cats]
        bars = ax.bar([xi + offset(i) for xi in x], vals, width,
                      label=AGENT_LABELS[agent], color=AGENT_COLORS[agent])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("avg total tokens per query")
    ax.set_title("Tokens per query by category (lower = better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tokens_by_category.png", dpi=140)
    plt.close()

    # 2. quality by category
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, agent in enumerate(AGENTS):
        vals = [summary["by_category"][c][agent].get("avg_quality") or 0 for c in cats]
        bars = ax.bar([xi + offset(i) for xi in x], vals, width,
                      label=AGENT_LABELS[agent], color=AGENT_COLORS[agent])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("avg judge score (1-5)")
    ax.set_ylim(0, 5.2)
    ax.set_title("Answer quality by category (higher = better)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "quality_by_category.png", dpi=140)
    plt.close()

    # 3. tokens vs quality scatter
    fig, ax = plt.subplots(figsize=(9, 5))
    for agent in AGENTS:
        pts = [(r["total_tokens"], r.get("quality_avg"))
               for (qid, ag), r in by_key.items() if ag == agent and r.get("quality_avg") is not None]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   c=AGENT_COLORS[agent], label=AGENT_LABELS[agent],
                   alpha=0.75, s=60, edgecolors="white")
    ax.set_xlabel("total tokens per query (log)")
    ax.set_ylabel("judge quality score (1-5)")
    ax.set_title("Tokens vs. quality — each point is one query")
    ax.set_xscale("log")
    ax.set_ylim(0, 5.2)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tokens_vs_quality.png", dpi=140)
    plt.close()

    # 4. tradeoff radar (3 measured + 3 qualitative axes)
    radar_data = _tradeoff_radar_data(summary)
    _draw_radar(radar_data, summary["overall"], out_dir / "tradeoff_radar.png")
    summary["radar_data"] = radar_data

    # 5. qualitative-axes heatmap (close-up of the 3 scored axes)
    _draw_qualitative_heatmap(out_dir / "qualitative_heatmap.png")
    summary["qualitative_scores"] = QUALITATIVE_SCORES

    # rewrite summary.json now that radar/qualitative fields are attached
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    import sys
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/runs.jsonl")
    judg = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("runs/judgments.jsonl")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("report")
    s = aggregate(runs, judg, out)
    print(json.dumps(s, indent=2))
