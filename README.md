# Wiki-fed agents vs. RAG — a real experiment on the sw30labs wiki

**TL;DR:** Four agents, one corpus, one judge. *Single-shot RAG* (FAISS top-k) vs. *agentic-RAG* (same FAISS, ≤8-turn tool loop) vs. *wiki navigation* (precomputed sitemap + page reads) vs. *QMD* (fully-local BM25 + on-device embeddings + Qwen3 reranker — "kind of like AI grep"). The earlier finding — *agentic-RAG is the Pareto winner: same quality as wiki-nav at ~40 % of the token cost* — still holds. The new finding comes from QMD: it's the **cheapest** and **tightest** retriever (~922 tokens/query, $0.005/query, **no embedding API**) and ties the other three on narrow factoid queries, but it **collapses on branch and cross-branch synthesis questions** (quality drops to 2.83 / 3.37 vs. 4.57+ for the others). QMD is a legitimate Pareto point for grep-shaped questions — and a good illustration of *why* an LLM-assisted retriever is not a drop-in for RAG when synthesis is required.

This repo is the reproducible source for those numbers.

> **Thanks to [Thomas De Souza](https://www.linkedin.com/in/ACoAAABPSp8BvNatjJFNW0LPxj2veG_xn_L9dpo)** for flagging [tobi/qmd](https://github.com/tobi/qmd) and framing it as *"kind of like AI grep"* — which turns out to be exactly the right framing, as the numbers below show.

- **Corpus:** `github.com/sw30labs/.github.wiki` pinned at commit `c130a3b` (89 markdown files, ~45 K corpus tokens)
- **Agent model:** Claude Sonnet 4.6 (temp 0) for all four pipelines
- **Single-shot RAG:** OpenAI `text-embedding-3-small` + FAISS IndexFlatIP, top-k = 5, section-aware chunking (800-token window, 150-token overlap), one Claude call
- **Agentic-RAG:** same FAISS index, wrapped in a ≤8-turn tool-call loop; tools are `retrieve(query, k)` and `read_full_page(path)`
- **Wiki navigation:** precomputed hierarchical sitemap (5 364 tokens) + `read_page(path)` / `list_children(prefix)` tools; ≤8 turns
- **QMD:** on-device hybrid retrieval — SQLite FTS5 BM25 + EmbeddingGemma vectors + Qwen3 reranker (all local via `@tobilu/qmd`), top-n = 5, single Claude call. Zero embedding API spend; ~2 GB GGUF model download amortized as setup cost.
- **Eval:** 30 synthesized day-to-day queries (10 narrow / 10 branch / 10 cross-branch) with ground-truth source paths
- **Judge:** Claude Opus 4.7 as LLM-judge, absolute 1–5 scoring on relevance + completeness + groundedness

## Results

### Headline numbers

| Metric                   |     RAG   | Agentic-RAG | Wiki nav  |   QMD    |
|--------------------------|----------:|------------:|----------:|---------:|
| Avg total tokens / query |    1,366  |       7,147 |   19,120  |   **922**|
| Avg cost / query         |  $0.0069  |    $0.0286  |  $0.0640  | **$0.0052** |
| Avg latency / query      |   5.4 s   |    13.7 s   |   11.7 s  |  10.1 s  |
| Avg turns                |    1.0    |     2.7     |    2.2    |    1.0   |
| Avg quality (1–5)        |   4.45    |   **4.77**  |    4.69   |    3.56  |

**Ratios vs. single-shot RAG:** Agentic-RAG costs 4.2× / tokens 5.2× / quality +0.32. Wiki-nav costs 9.3× / tokens 14.0× / quality +0.24. QMD costs **0.76×** / tokens **0.67×** / quality **−0.89**.

### Head-to-head (quality wins over 30 queries, ties within ±0.17 on 1–5 scale)

| Pairing                     |  A wins | B wins | Ties | Mean Δ (B − A) |
|-----------------------------|--------:|-------:|-----:|---------------:|
| RAG  vs  Agentic-RAG        |       2 |  **11**|   17 |     +0.32      |
| RAG  vs  Wiki nav           |       7 |  **11**|   12 |     +0.24      |
| RAG  vs  QMD                |  **18** |      4 |    8 |     −0.89      |
| Agentic-RAG  vs  Wiki nav   |  **10** |      6 |   14 |     −0.08      |
| Agentic-RAG  vs  QMD        |  **24** |      0 |    6 |     −1.21      |
| Wiki nav  vs  QMD           |  **22** |      2 |    6 |     −1.13      |

Two findings land at once:

1. **Agentic-RAG remains the single Pareto winner on quality-vs-cost** — same quality as wiki (−0.08, inside noise) at 45 % of the cost, and beats single-shot RAG 11–2 for 4.2× the per-query spend.
2. **QMD is the cheapest arm but loses ~90 % of its head-to-head matchups vs. agentic-RAG (24–0, 6 ties).** On the 30-query set it scored 0 wins against agentic-RAG. Its "AI grep" framing is accurate — it's a great structured-search tool, not a RAG replacement.

### By query category

| Category     | RAG tok | Agentic tok | Wiki tok | QMD tok | RAG q | Agentic q | Wiki q |  QMD q  |
|--------------|--------:|------------:|---------:|--------:|------:|----------:|-------:|--------:|
| narrow       |  1,096  |       3,028 |   11,727 |     800 |  5.00 |      5.00 |   4.93 |  4.47   |
| branch       |  1,578  |       8,808 |   22,034 |     952 |  4.43 |      4.60 |   4.57 | **2.83**|
| cross-branch |  1,423  |       9,604 |   23,599 |   1,014 |  3.90 |      4.70 |   4.57 | **3.37**|

QMD holds up on **narrow** factoid lookups (quality 4.47, tied with the other three within a rounding error), but its quality **collapses** on branch (2.83) and cross-branch (3.37) queries where the other three sit at 4.5+. Two concrete failure modes surface when you read the run log:

- **Compact-snippet trap.** QMD returns short reranked snippets, not full chunks. On `q04` ("What TTS model does the audiobook generator use?") it *retrieved the correct file* but the snippet shown to Claude was the front-matter (`id`, `name`, `repo_path`) with none of the body — so Claude answered "the passages don't contain the answer." RAG, agentic-RAG, and wiki-nav all scored 5.00 on the same query because they saw the full chunk body.
- **Missing landing pages.** On `q17` ("Which of our agentic frameworks are for LangGraph coding assistants?"), QMD's BM25+reranker surfaced peripheral stubs (`agent-stack-stub`, `sitemap.md`, unrelated OSCAL/pentest files) and missed `wiki/agentic-frameworks/agentic-frameworks.md` — the obvious branch landing page that all other three retrieved cleanly.

## Example queries

Three queries from the run, one per category, showing where each agent's retrieval shape starts to matter. Full per-query records in `runs/runs.jsonl` and `runs/judgments.jsonl`.

### `q00` — narrow, where all four agents tie

> **Which tool extracts critical CVEs with network attack vectors from the NVD API?**
>
> Ground truth: `raw/nvd-extractor.stub.md`

| Agent        | Quality | Tokens | Cost    | Notes |
|--------------|--------:|-------:|--------:|-------|
| RAG          |    5.00 |  1,242 | $0.0054 | One call, right answer. |
| Agentic-RAG  |    5.00 |  3,261 | $0.0123 | Same answer, 2 turns. |
| Wiki nav     |    4.67 |  7,465 | $0.0239 | Answered from the sitemap summary without opening a page. |
| QMD          |    5.00 |    828 | $0.0038 | **Cheapest correct answer on the whole eval.** BM25+rerank finds `nvd-extractor.stub.md` cleanly; tight snippet is enough. |

On narrow factoid questions the retrieval substrate barely matters — all four tie and **QMD is ~30 % cheaper than single-shot RAG**.

### `q24` — cross-branch, where single-shot RAG breaks but QMD survives

> **If I want to build a digital-twin-based incident response simulation, which article frames the concept and which OSCAL repo implements the drift detection?**
>
> Ground truth: `wiki/articles/generative-ai-digital-twins-for-proactive-and.md`, `raw/oscal-digital-twin-playground.stub.md`, `wiki/oscal-compliance/oscal-compliance.md`

| Agent        | Quality | Tokens | Cost    | Notes |
|--------------|--------:|-------:|--------:|-------|
| RAG          |    3.33 |  1,408 | $0.0073 | Missed the article; hedged. Top-5 FAISS chunks didn't surface it. |
| Agentic-RAG  |    5.00 |  4,253 | $0.0187 | Refined after first retrieve, got both artefacts. |
| Wiki nav     |    5.00 | 16,858 | $0.0569 | Same answer as agentic-RAG; 4× the tokens. |
| QMD          |    4.67 |    979 | $0.0063 | Reranker found both target docs; compact snippets happened to carry the key facts. |

On this one, QMD's reranker actually beats single-shot RAG — a reminder that the problem isn't *local hybrid retrieval*, it's *compact snippets* when Claude needs full-chunk substrate.

### `q17` — branch, where QMD collapses

> **For someone building coding assistants with LangGraph, which of our agentic frameworks are most relevant and how do they differ?**
>
> Ground truth: `wiki/agentic-frameworks/agentic-frameworks.md`

| Agent        | Quality | Tokens | Cost    | Notes |
|--------------|--------:|-------:|--------:|-------|
| RAG          |    4.33 |  2,342 | $0.0098 | Surfaced `agentic-frameworks.md` via top-5; answer is correct but compressed. |
| Agentic-RAG  |    4.33 | 16,712 | $0.0616 | Read multiple pages; answer is thorough. |
| Wiki nav     |    4.67 | 16,420 | $0.0548 | Sitemap pointed straight at the branch; one page read sufficed. |
| QMD          |  **1.33** |    898 | $0.0050 | Reranker surfaced peripheral stubs + the sitemap; **missed the actual branch landing page**. Claude correctly said "can't identify which specific repos" — an honest failure, not a hallucination, but a failure nonetheless. |

This is the core QMD failure mode: on questions that require *starting from the hierarchy* (a branch landing page like `wiki/agentic-frameworks/agentic-frameworks.md`), QMD's lexical+semantic ranking scores leaf stubs and sitemap fragments higher than the landing document itself.

## Scope boundaries (important)

**What this experiment measures:** per-query, day-to-day agent cost and answer quality.

**What it deliberately excludes** (per the brief):
- Building the wiki itself (one-time curator effort)
- Building the RAG index (one-time; logged separately — $0.0009 for this corpus, 9.1 s)
- Building the QMD collection (local, $0 API; ~2 GB GGUF model download and ~9 s local compute — logged in `qmd/index_meta.json`)

Setup costs are one-offs any production team pays once; they're out of scope for a steady-state comparison. QMD's "no API spend per query" is real but comes with a one-time local-compute footprint the table above doesn't show.

## What the numbers don't measure

Per-query cost and quality are only part of the picture. Three axes the benchmark ignores but a production decision shouldn't:

- **Operational surface.** Wiki-nav is a sitemap parser plus a two-tool loop — markdown files in git, nothing else underneath. RAG is a chunker, an embedder, a vector store, and a loop, with tuning knobs (chunk size, top-k, embed model) each needing their own evaluation. QMD is self-contained (no embedding API, no vector store to maintain) but adds a Bun runtime and ~2 GB of GGUF models to the deployment surface. Wiki-nav still survives anything `git clone` + an LLM client survives.
- **Dual-surface value.** A wiki is a human-readable product on day one — GitHub renders it, people browse it, contributors open PRs against it. A FAISS or QMD index is machine-only; to give humans the same access you build a second product.
- **Incremental updates.** Wiki update = edit markdown, `git push`, done. RAG update = re-chunk, re-embed, re-upsert. QMD update = `qmd update` (cheaper because re-embedding is local, but still a step beyond `git push`).

Agentic-RAG wins on $/query. QMD wins on $/query *for grep-shaped questions only*. Neither wins when you add the cost of *building* their stack, *maintaining* the index, and *separately* building the human-readable surface the wiki gave you for free.

### Tradeoff shape — the whole picture on one chart

![tradeoff radar — measured and qualitative axes overlaid](report/tradeoff_radar.png)

Three measured axes (Quality, Cost efficiency, Latency) min-max normalized across the four agents, and three qualitative axes (Operational simplicity, Human-readable surface, Update velocity) author-scored 0–1. Outer = better on every axis.

Each pipeline occupies a different region:

- **RAG (orange)** — strong on Cost + Latency, collapsed on the qualitative axes.
- **Agentic-RAG (green)** — spike on Quality, mid-tier on Cost, worst on Latency, collapsed on qualitative.
- **Wiki nav (blue)** — dominates the qualitative half of the chart, moderate Quality, weak Cost + Latency.
- **QMD (violet)** — spike on Cost efficiency (cheapest measured), bottom on Quality, moderate on the qualitative axes (local → no embedding-API dependency, but still machine-only).

Raw values behind the normalized axes: Quality — RAG 4.45 / Agentic 4.77 / Wiki 4.69 / QMD 3.56. Cost/query — $0.0069 / $0.0286 / $0.0640 / $0.0052. Latency — 5.4 s / 13.7 s / 11.7 s / 10.1 s.

![qualitative axes heatmap](report/qualitative_heatmap.png)

Close-up of the three qualitative axes. RAG and agentic-RAG share the FAISS / embedding substrate and score identically. QMD scores mid-tier on operational simplicity (Bun runtime + GGUF models, but no external API) and update velocity (local re-embed is cheap), and identical to RAG on human-readable surface (SQLite index, not human-browseable).

## Honest caveats

Adding agentic-RAG closed the single-shot-vs-agentic confound; adding QMD tests whether local hybrid retrieval can match cloud-embed RAG. Both findings have the following caveats:

- **Evaluation circularity.** Queries were synthesized by Opus *with the sitemap in its context* and ground-truth source paths were produced by the same synthesis call. The sitemap-primed synthesizer plausibly biases toward questions the wiki agent can answer cleanly. This isn't just bias — it's a leak. A follow-up with a sitemap-blind synthesizer would tighten the finding, especially for QMD (which may be disadvantaged by questions structured around the wiki's hierarchy).
- **QMD apples-to-oranges.** QMD's retriever is a 3-stage pipeline (BM25 + vector + reranker + HyDE query expansion); the RAG arm is vanilla top-5 FAISS with no reranker. QMD's quality loss is therefore *about its snippet shape*, not about local-vs-cloud embeddings — a QMD run that returned full chunks instead of reranker-style snippets would likely close most of the branch/cross-branch gap. This experiment doesn't disentangle those knobs.
- **QMD also runs a single-shot loop.** An agentic-QMD arm (QMD's `read` + multi-turn refine) would test whether the problem is local retrieval or the single-shot format; not included here.
- **Statistical rigor.** n = 10 per category, single run, single seed, single judge (Claude judging Claude). No confidence intervals, no bootstrap, no significance tests. The gap between agentic-RAG and single-shot RAG (+0.32) is real; the gap between agentic-RAG and wiki (−0.08) is inside likely-noise; the gap between QMD and the other three is large enough (−0.89 to −1.21) to be directional.
- **Corpus-specific.** The sw30labs wiki is small (45 K tokens) and well-structured. Sprawling, unstructured corpora would likely favor RAG variants further; corpora with rich taxonomy might favor wiki; corpora where most questions are lexical lookups (API references, error-code catalogs) might favor QMD.
- **Implementation-specific.** The RAG stack is vanilla `text-embedding-3-small` + top-k FAISS. No reranker, no hybrid BM25, no query expansion on the RAG side. Most of agentic-RAG's remaining loss to wiki on the hardest queries likely lives here. This experiment does not test those knobs.
- **No failure taxonomy.** Point-estimate quality only; no breakdown of *why* each arm lost the ones it lost (hallucination vs. retrieval miss vs. incomplete citation), no p95 latency, no separate hallucination-rate metric, no cost-per-*correct*-answer.
- **Single model family.** Judge and agents are all Anthropic; Claude-judging-Claude has known intra-family preferences. A second-judge cross-check (e.g., GPT-4 or a human subset) is not included.
- **Hardware-specific latency.** QMD's 10.1 s average latency reflects local inference on an M-series Mac. On x86 Linux servers without Metal acceleration the numbers will look different.
- **Operational concerns unmodeled.** Throughput / rate-limit shape, multi-tenant ACL filtering, sitemap maintenance decay, update-latency when pages change — all out of scope.

## What to do with this

If the production decision is RAG-or-wiki for a day-to-day assistant over a wiki-shaped corpus:

- **Default answer: wrap RAG in an agentic loop.** Same quality as wiki-nav at 45 % of the cost; clear winner on the single Pareto frontier the experiment actually measures.
- **If your queries are mostly lookups, try QMD first.** On narrow/factoid questions QMD ties the other three at ~30 % of single-shot RAG's cost and zero embedding API spend. "AI grep" is exactly what it is and exactly when it's good.
- **Wiki-nav still wins on trace readability.** Sitemap + page reads is easier to explain to a skeptical reviewer than embedding hits or reranker scores; and it may win on corpora where titles and hierarchy carry most of the signal.

Three natural next experiments:
1. **Hybrid routing.** Wiki picks the subtree, RAG retrieves inside it. Should drop the sitemap cost while preserving agentic-RAG's quality.
2. **QMD-with-full-chunks.** Switch QMD from reranker-snippet output to full chunk bodies; test whether the branch/cross-branch collapse is about the snippet shape or about the retriever itself.
3. **Agentic-QMD.** QMD exposes an MCP server; wrap it in the same ≤8-turn loop as agentic-RAG and test whether local hybrid retrieval + turn budget closes the synthesis gap.

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

# 4. one-time QMD bootstrap (installs @tobilu/qmd; ~2 GB GGUF models download on first qmd-index run)
make qmd-setup   # requires Bun — https://bun.sh

# 5. run everything (~10 min, ~$8 of API spend)
make all
```

Or run phases individually: `make index | make sitemap | make qmd-index | make queries | make eval | make judge | make report`.

All raw per-query logs are in `runs/runs.jsonl` and `runs/judgments.jsonl`. Every number above can be recomputed from those files via `report/aggregate.py`.

## Repo layout

```
wiki-vs-rag/
├── README.md                   (this file)
├── Makefile                    (make all / make eval / make report)
├── corpus/                     (wiki pinned at c130a3b)
├── rag/
│   ├── chunker.py              (section-aware markdown chunker)
│   ├── indexer.py              (one-time: embed + build FAISS)
│   └── agent.py                (query-time: embed → top-k → one Claude call)
├── rag_agentic/
│   └── agent.py                (query-time: Claude + retrieve/read_full_page tools, ≤8 turns)
├── wiki/
│   ├── sitemap.py              (one-time: build hierarchical map)
│   └── agent.py                (query-time: Claude + read_page/list_children tools, ≤8 turns)
├── qmd/
│   ├── indexer.py              (one-time: stage corpus + qmd collection add + qmd embed)
│   └── agent.py                (query-time: subprocess qmd query → one Claude call)
├── eval/
│   ├── synthesize.py           (Opus 4.7 generates 30 queries + ground truths)
│   ├── run.py                  (runs all three pipelines over the query set)
│   ├── judge.py                (Opus 4.7 judge, absolute 1-5 scoring)
│   └── queries.jsonl           (the 30 queries — committed for reproducibility)
├── runs/                       (raw per-query logs)
│   ├── runs.jsonl
│   └── judgments.jsonl
└── report/
    ├── summary.json            (aggregate stats)
    ├── results.csv             (row per (query, agent))
    └── *.png                   (charts)
```

## Cost of running this experiment

| Phase                                | Cost       |
|--------------------------------------|-----------:|
| RAG index build (one-time)           | $0.0009    |
| Sitemap build (no LLM)               | $0.0000    |
| QMD index build (local, no API)      | $0.0000 *(+ ~2 GB GGUF download, ~9 s local compute)* |
| Query synthesis (Opus 4.7 × 30)      | $1.35      |
| Eval runs — RAG (Sonnet × 30)        | $0.21      |
| Eval runs — Agentic-RAG (Sonnet × 30)| $0.86      |
| Eval runs — Wiki (Sonnet × 30)       | $1.92      |
| Eval runs — QMD (Sonnet × 30)        | $0.16      |
| Judge (Opus 4.7 × 120)               | $5.26      |
| **Total**                            | **~$9.76** |

## Author

Nicolas Cravino · [sw30labs](https://github.com/sw30labs) · 2026-04-20

*The QMD arm was added after [Thomas De Souza](https://www.linkedin.com/in/ACoAAABPSp8BvNatjJFNW0LPxj2veG_xn_L9dpo) suggested [tobi/qmd](https://github.com/tobi/qmd) — "kind of like AI grep" — as an interesting fourth benchmark. He was right on both counts.*
