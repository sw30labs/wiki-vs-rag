"""LLM-judge for answer quality (absolute scoring).

For each (query, agent) pair, score the answer independently on
relevance, completeness, and groundedness (1-5 each). The judge sees
the ground-truth source content for the query and one answer at a time.

Absolute scoring (vs. pairwise A/B) eliminates A/B ordering bias and
scales to >2 agents without combinatorial judge calls. Head-to-head
tallies are computed downstream in report/aggregate.py.
"""
from __future__ import annotations
import os, json
from pathlib import Path
import anthropic

MODEL = "claude-opus-4-7"
OPUS_IN = 15.0
OPUS_OUT = 75.0

JUDGE_PROMPT = """You are a rigorous evaluator scoring one answer to a question about the sw30labs organization wiki.

Question:
{query}

Ground-truth source content (the authoritative material the answer should be grounded in):
{truth}

Answer to evaluate:
{answer}

Score the answer on three axes, 1-5 integer:
- relevance: does the answer address the actual question?
- completeness: does the answer cover the key facts needed?
- groundedness: are the factual claims supported by the ground-truth sources? (penalize hallucinations hard)

Return ONLY valid JSON (no fences):
{{"relevance": N, "completeness": N, "groundedness": N, "notes": "one line"}}
"""


def _truth_text(corpus: Path, paths: list[str], max_chars: int = 6000) -> str:
    out = []
    for p in paths:
        fp = corpus / p
        if fp.exists():
            body = fp.read_text(encoding="utf-8", errors="ignore")
            out.append(f"--- {p} ---\n{body}")
    joined = "\n\n".join(out)
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n... [truncated for judge prompt]"
    return joined


def run_judge(runs_path: Path, corpus_root: Path, out_path: Path) -> dict:
    records = [json.loads(l) for l in runs_path.read_text().splitlines()]

    ac = anthropic.Anthropic()
    out_f = open(out_path, "w")
    usage = {"input": 0, "output": 0}
    n_scored = 0
    n_failed = 0

    for r in records:
        qid = r["id"]
        agent = r["agent"]
        truth = _truth_text(corpus_root, r["ground_truth"])
        prompt = JUDGE_PROMPT.format(query=r["query"], truth=truth, answer=r["answer"])

        resp = ac.messages.create(
            model=MODEL, max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        usage["input"] += resp.usage.input_tokens
        usage["output"] += resp.usage.output_tokens
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            j = json.loads(text)
        except Exception as e:
            n_failed += 1
            print(f"  JUDGE PARSE FAIL {qid}/{agent}: {e}\n{text[:200]}")
            continue

        rec = {
            "id": qid,
            "category": r["category"],
            "agent": agent,
            "scores": {
                "relevance": j.get("relevance"),
                "completeness": j.get("completeness"),
                "groundedness": j.get("groundedness"),
                "notes": j.get("notes", ""),
            },
        }
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        n_scored += 1
        avg = (j["relevance"] + j["completeness"] + j["groundedness"]) / 3
        print(f"[{qid}] {agent}: {avg:.2f}")

    out_f.close()
    cost = usage["input"] / 1_000_000 * OPUS_IN + usage["output"] / 1_000_000 * OPUS_OUT
    summary = {
        "judge_input_tokens": usage["input"],
        "judge_output_tokens": usage["output"],
        "judge_cost_usd": round(cost, 4),
        "model": MODEL,
        "n_scored": n_scored,
        "n_parse_failures": n_failed,
    }
    (out_path.parent / "judge_meta.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/runs.jsonl")
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("runs/judgments.jsonl")
    print(json.dumps(run_judge(runs, corpus, out), indent=2))
