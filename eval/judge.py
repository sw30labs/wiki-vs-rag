"""Blind LLM-judge for answer quality.

For each query, present both answers in RANDOMIZED ORDER (A/B) labeled only
by letter. Judge is given the ground-truth source paths' content.
Scores 1-5 on relevance, completeness, groundedness.
"""
from __future__ import annotations
import os, json, random
from pathlib import Path
import anthropic

MODEL = "claude-sonnet-4-6"  # cost-conscious choice; caveat noted in README
SONNET_IN = 3.0
SONNET_OUT = 15.0

JUDGE_PROMPT = """You are a rigorous, blind evaluator comparing two answers to the same question about the sw30labs organization wiki.

Question:
{query}

Ground-truth source content (the authoritative material the answer should be grounded in):
{truth}

Answer A:
{answer_a}

Answer B:
{answer_b}

Score each answer on three axes, 1-5 integer:
- relevance: does the answer address the actual question?
- completeness: does the answer cover the key facts needed?
- groundedness: are the factual claims supported by the ground-truth sources? (penalize hallucinations hard)

Return ONLY valid JSON (no fences):
{{"A": {{"relevance": N, "completeness": N, "groundedness": N, "notes": "one line"}},
  "B": {{"relevance": N, "completeness": N, "groundedness": N, "notes": "one line"}}}}
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


def run_judge(runs_path: Path, corpus_root: Path, out_path: Path, seed: int = 7) -> dict:
    records = [json.loads(l) for l in runs_path.read_text().splitlines()]
    # group by query id
    by_id: dict[str, dict] = {}
    for r in records:
        by_id.setdefault(r["id"], {})[r["agent"]] = r

    rng = random.Random(seed)
    ac = anthropic.Anthropic()
    out_f = open(out_path, "w")
    usage = {"input": 0, "output": 0}

    for qid, pair in by_id.items():
        if "rag" not in pair or "wiki" not in pair:
            continue
        r = pair["rag"]
        w = pair["wiki"]
        # randomize ordering
        flip = rng.random() < 0.5
        a, b = (w, r) if flip else (r, w)
        a_agent, b_agent = ("wiki", "rag") if flip else ("rag", "wiki")

        truth = _truth_text(corpus_root, r["ground_truth"])
        prompt = JUDGE_PROMPT.format(
            query=r["query"], truth=truth,
            answer_a=a["answer"], answer_b=b["answer"],
        )

        resp = ac.messages.create(
            model=MODEL, max_tokens=500, temperature=0,
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
            print(f"  JUDGE PARSE FAIL {qid}: {e}\n{text[:200]}")
            continue

        rec = {
            "id": qid,
            "category": r["category"],
            "a_agent": a_agent,
            "b_agent": b_agent,
            "A": j.get("A", {}),
            "B": j.get("B", {}),
        }
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        a_avg = (j["A"]["relevance"] + j["A"]["completeness"] + j["A"]["groundedness"]) / 3
        b_avg = (j["B"]["relevance"] + j["B"]["completeness"] + j["B"]["groundedness"]) / 3
        print(f"[{qid}] A={a_agent}:{a_avg:.1f}  B={b_agent}:{b_avg:.1f}")

    out_f.close()
    cost = usage["input"] / 1_000_000 * SONNET_IN + usage["output"] / 1_000_000 * SONNET_OUT
    summary = {
        "judge_input_tokens": usage["input"],
        "judge_output_tokens": usage["output"],
        "judge_cost_usd": round(cost, 4),
        "model": MODEL,
    }
    (out_path.parent / "judge_meta.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/runs.jsonl")
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus")
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("runs/judgments.jsonl")
    print(json.dumps(run_judge(runs, corpus, out), indent=2))
