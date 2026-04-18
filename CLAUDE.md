# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single-purpose, reproducible benchmark comparing two agent patterns over the `sw30labs/.github.wiki` corpus (pinned at commit `c130a3b` under `corpus/`):

- **RAG**: embed query → FAISS top-k → one Claude call with retrieved chunks in context
- **Wiki navigation**: Claude sees a precomputed sitemap up-front and uses `read_page` / `list_children` tools to navigate

Everything is a phased pipeline driven by `Makefile`. Outputs (indices, sitemaps, runs, judgments, charts) are checked in so partial reruns work.

## Commands

```bash
pip install -r requirements.txt           # openai, anthropic, faiss-cpu, tiktoken, pyyaml, numpy, matplotlib
source .env                                # sets OPENAI_API_KEY, ANTHROPIC_API_KEY

make all                                   # full pipeline (~7 min, ~$3-4 of API spend)

# individual phases — each target guards its output file, so `make <phase>` is idempotent
make index     # rag/index/index_meta.json        (build FAISS from corpus/)
make sitemap   # wiki/sitemap/sitemap_meta.json   (build hierarchical sitemap.md)
make queries   # eval/queries.jsonl               (Opus synthesizes 30 queries)
make eval      # runs/runs.jsonl                  (run both agents over queries)
make judge     # runs/judgments.jsonl             (blind A/B judge)
make report    # report/summary.json + PNGs + CSV

make clean     # wipe derived artefacts (keeps eval/queries.jsonl)
make reset     # clean + also delete the synthesized query set
```

Each Python module is also runnable standalone with positional args — see each file's `if __name__ == "__main__":` block. Useful for spot-checking a single query:

```bash
python3 rag/agent.py "What repos are in OSCAL & Compliance?" rag/index
python3 wiki/agent.py "..." corpus wiki/sitemap/sitemap.md
```

There is no test suite, linter, or type checker configured. Python 3 only; no virtualenv or lockfile.

## Architecture

The pipeline has two independent setup paths that converge at `eval/run.py`:

```
corpus/ ──► rag/chunker.py ──► rag/indexer.py ──► rag/index/       ─┐
                                                                    ├──► eval/run.py ──► runs/runs.jsonl ──► eval/judge.py ──► runs/judgments.jsonl ──► report/aggregate.py ──► report/
corpus/ ──► wiki/sitemap.py ──► wiki/sitemap/sitemap.md ────────────┤
                     │                                              │
                     └──► eval/synthesize.py ──► eval/queries.jsonl ┘
```

### Key design rules baked into the code

- **Setup vs. per-query separation is load-bearing.** The whole experiment's thesis rests on excluding one-time costs from per-query metrics. `rag/indexer.py` and `wiki/sitemap.py` run once and their costs are logged to `*_meta.json`. The per-query agents in `rag/agent.py` and `wiki/agent.py` only measure the query-time call. Don't fold setup cost into the per-query measurement.
- **Sitemap summaries are not LLM-generated.** `wiki/sitemap.py` pulls the first non-header sentence (or front-matter `title`) deliberately — generating summaries with an LLM would leak LLM cost into setup. If you change this, update the README caveat.
- **Both agents share the same model and temperature** (`claude-sonnet-4-6`, temp 0, `max_tokens=700`) for a fair comparison. The constants live at the top of each agent file. Keep them in sync across `rag/agent.py` and `wiki/agent.py`.
- **Pricing is hardcoded** (per-1M: Sonnet $3 in / $15 out; Opus $15/$75; embeddings $0.02) near the top of each file that computes `cost_usd`. Update all sites together if prices change.
- **Token counting uses `tiktoken` `cl100k_base`** for corpus/sitemap measurement; Anthropic's `usage.input_tokens` / `usage.output_tokens` are the ground truth for per-query cost.
- **Judge randomizes A/B ordering** (`eval/judge.py`, seeded `random.Random(7)`) and `report/aggregate.py` de-blinds via the `a_agent`/`b_agent` fields. Don't break this contract.
- **Head-to-head ties** in `report/aggregate.py` are defined as `|Δquality| < 0.17` (≈0.5 on any single axis). Baked-in threshold — change deliberately.

### RAG pipeline details

- `rag/chunker.py` splits markdown on H1/H2/H3 headers, then falls back to an 800-token sliding window with 150-token overlap. Files whose name starts with `_` (e.g. `_Sidebar.md`, `_Footer.md`) are skipped everywhere — the chunker, the sitemap, and `list_children`.
- `rag/indexer.py` uses `text-embedding-3-small` with batch=96, FAISS `IndexFlatIP` with L2-normalized vectors (so inner-product == cosine).
- `rag/agent.py` retrieves top-5 chunks and packs them into a single Claude call with source-path citations required.

### Wiki pipeline details

- `wiki/agent.py` runs a tool-call loop capped at `MAX_TURNS=8`. If the cap is hit, it does a final no-tools turn asking for an answer with what's been read — this counts toward token usage and is flagged via `hit_turn_cap`.
- Tool dispatcher in `wiki/agent.py` has a path-traversal guard in `_read_page`; keep it if you refactor.
- The sitemap is injected into the system prompt on every turn, so the ~5k-token sitemap is billed once per turn, not once per query — this is the dominant cost driver and the reason prompt caching is listed in the README as a natural optimization.

### Eval and reporting

- `eval/synthesize.py` calls Opus in three batches (narrow/branch/cross-branch, 10 each). Each query carries ground-truth `sources` paths used later by the judge. Queries are committed to `eval/queries.jsonl` for reproducibility; regenerating them is non-deterministic (temp 0.4).
- `eval/run.py` has a `BUDGET_USD = 10.0` circuit breaker that aborts mid-run if exceeded.
- `eval/judge.py` tolerates one JSON parse failure per query (prints and skips). `report/aggregate.py` therefore must handle missing quality scores — don't assume every `(id, agent)` pair has a `quality_avg`.
- `report/aggregate.py` writes `summary.json`, `results.csv`, and three PNGs. The PNGs are referenced from the README — regenerate them if you change chart code.

## Conventions

- Paths are stored as **relative to `corpus/`** everywhere (chunks, sitemap entries, ground truths, tool inputs, citations). Don't mix absolute paths in.
- All derived artefacts land under the module that produced them (`rag/index/`, `wiki/sitemap/`, `runs/`, `report/`). `make clean` knows these locations — add new outputs to the `clean` target if you introduce more.
- `_private/` exists but is not part of the experiment; treat as out-of-scope.
