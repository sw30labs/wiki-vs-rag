# Wiki-fed agents vs. RAG — a real experiment on the sw30labs wiki

**TL;DR:** The common intuition that wiki-fed agents are *token-efficient* compared to RAG is **wrong for this corpus and this implementation**. Wiki navigation uses **~15× more tokens and costs ~10× more per query**. What it buys you is **meaningfully higher answer quality (+0.56 points on a 1–5 scale; 19W–3L–7T head-to-head)**, with the largest quality gap on hard cross-branch queries. The real trade-off is **cost vs. answer quality**, not efficiency.

This repo is the reproducible source for those numbers.

- **Corpus:** `github.com/sw30labs/.github.wiki` pinned at commit `c130a3b` (90 markdown files, ~45 K corpus tokens)
- **Agent model:** Claude Sonnet 4.6 (temp 0) for both pipelines
- **RAG stack:** OpenAI `text-embedding-3-small` + FAISS IndexFlatIP, top-k = 5, section-aware chunking (800-token window, 150-token overlap)
- **Wiki stack:** precomputed hierarchical sitemap (5 364 tokens) + two tools: `read_page(path)`, `list_children(prefix)`; max 8 tool-call turns
- **Eval:** 30 synthesized day-to-day queries (10 narrow / 10 branch / 10 cross-branch) with ground-truth source paths
- **Judge:** Claude Sonnet 4.6 as blind LLM-judge, randomized A/B ordering, scoring relevance + completeness + groundedness (1–5 each)

## Results

### Headline numbers

| Metric                    | RAG        | Wiki navigation | Ratio (Wiki / RAG) |
|---------------------------|------------|-----------------|--------------------|
| Avg total tokens / query  | **1 370**  | **20 308**      | **14.8×**          |
| Avg cost / query          | **$0.0069**| **$0.0687**     | **9.9×**           |
| Avg latency / query       | 4.8 s      | 12.8 s          | 2.7×               |
| Avg turns                 | 1.0        | 2.3             | —                  |
| Avg quality (1–5)         | **3.92**   | **4.48**        | +0.56              |
| Head-to-head quality wins | 3          | 19              | (7 ties)           |

### By query category

| Category     | RAG tokens | Wiki tokens | RAG quality | Wiki quality |
|--------------|-----------:|------------:|------------:|-------------:|
| narrow       |      1 206 |      16 824 |       4.54  |        4.70  |
| branch       |      1 508 |      19 932 |       4.10  |        4.60  |
| cross-branch |      1 395 |      24 168 |       3.04  |        4.11  |

The quality gap is tiny on narrow queries (+0.16) and grows as the question spans more of the wiki, reaching +1.07 on cross-branch. **This is the most important finding of the experiment.**

![tokens by category](report/tokens_by_category.png)
![quality by category](report/quality_by_category.png)
![tokens vs quality scatter](report/tokens_vs_quality.png)

### What drives wiki's token cost

The 5 364-token sitemap is paid on **every** query as baseline input. The wiki agent then loads one or more whole pages (averaging ~2 000 tokens each) via tool calls — each tool result gets added to the next turn's input. With a mean of 2.3 turns and 2.0 pages read per query, input tokens dominate.

RAG, by contrast, retrieves 5 small section-aware chunks (~80 tokens each, ~400 tokens total) and makes a single call.

## Scope boundaries (important)

**What this experiment measures:** per-query, day-to-day agent cost and answer quality.

**What it deliberately excludes** (per the brief):
- Building the wiki itself (one-time curator effort)
- Building the RAG index (one-time; logged separately — $0.0009 for this corpus, 9.7 s)

Setup costs are one-offs any production team pays once; they're out of scope for a steady-state comparison.

## Honest caveats

- **Corpus-specific.** The sw30labs wiki is small (45 K tokens) and beautifully structured. Well-structured wikis flatter RAG on narrow queries (retrieval finds the one right chunk instantly) and flatter wiki on cross-branch ones (the map lets the agent reason about relationships). A sprawling, unstructured corpus would likely favor RAG even more.
- **Implementation-specific.** The wiki agent here is the **naive** pattern: full sitemap up-front, whole-page reads. Several optimizations are on the table (see below) that would close the token gap — none were implemented for this experiment.
- **Single model.** Results may shift with smaller/cheaper models or different providers.
- **LLM-as-judge has biases** even with blinded A/B ordering. A human cross-graded subset would strengthen the quality numbers; this experiment does not include one.
- **Judge uses Sonnet, not Opus.** A cheaper and slightly weaker judge. Running the same scoring with Opus is one line-change in `eval/judge.py`.
- **One parse failure.** Query `q28`'s judge output didn't parse as JSON (Sonnet emitted a preamble). 29 of 30 queries contribute to quality stats.

## Optimizations that would change the verdict

These are **not** implemented here but are the obvious next steps if token cost is the blocker:

1. **Prompt caching for the sitemap.** Anthropic's prompt caching (`cache_control`) would amortize the 5 364-token sitemap across a warm session. In a long-lived assistant this alone could cut wiki input tokens by ~30 %.
2. **Two-level sitemap.** Show only categories up front; have the agent call `list_children(category)` on demand. More turns, less input per turn.
3. **Section-level reads.** Replace `read_page(path)` with `read_section(path, heading)`. Closes most of the gap — but also starts to look a lot like RAG.
4. **Hybrid: wiki for routing, RAG inside the chosen branch.** The real productionization. The wiki chooses which subtree; RAG retrieves within it. Reduces sitemap overhead *and* scales to large corpora.

A follow-up experiment comparing these variants would be more informative than tuning either side in isolation.

## Reproduction

```bash
# 1. clone
git clone <this-repo>
cd wiki-vs-rag

# 2. install
pip install -r requirements.txt   # openai anthropic faiss-cpu tiktoken pyyaml numpy matplotlib

# 3. set keys
cp .env.example .env  && $EDITOR .env   # fill in OPENAI_API_KEY, ANTHROPIC_API_KEY
source .env

# 4. run everything (~7 min, ~$3-4 of API spend)
make all
```

Or run phases individually: `make index | make sitemap | make queries | make eval | make judge | make report`.

All raw per-query logs are in `runs/runs.jsonl` and `runs/judgments.jsonl`. Every number above can be recomputed from those files via `report/aggregate.py`.

## Repo layout

```
wiki-vs-rag/
├── README.md                  (this file)
├── Makefile                   (make all / make eval / make report)
├── corpus/                    (wiki pinned at c130a3b)
├── rag/
│   ├── chunker.py             (section-aware markdown chunker)
│   ├── indexer.py             (one-time: embed + build FAISS)
│   └── agent.py               (query-time: embed → top-k → Claude)
├── wiki/
│   ├── sitemap.py             (one-time: build hierarchical map)
│   └── agent.py               (query-time: Claude + read_page/list_children tools)
├── eval/
│   ├── synthesize.py          (Opus generates 30 queries + ground truths)
│   ├── run.py                 (runs both pipelines over the query set)
│   ├── judge.py               (blind LLM-judge, randomized A/B)
│   └── queries.jsonl          (the 30 queries — committed for reproducibility)
├── runs/                      (raw per-query logs)
│   ├── runs.jsonl
│   └── judgments.jsonl
└── report/
    ├── summary.json           (aggregate stats)
    ├── results.csv            (row per (query, agent))
    └── *.png                  (charts)
```

## Cost of running this experiment

| Phase                          | Cost       |
|--------------------------------|-----------:|
| RAG index build (one-time)     | $0.0009    |
| Sitemap build (no LLM)         | $0.0000    |
| Query synthesis (Opus × 30)    | $1.03      |
| Eval runs (Sonnet × 60)        | $2.27      |
| Judge (Sonnet × 30)            | $0.30      |
| **Total**                      | **$3.60**  |

## Author

Nicolas Cravino · [sw30labs](https://github.com/sw30labs) · 2026-04-18
