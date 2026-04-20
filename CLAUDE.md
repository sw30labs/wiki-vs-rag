# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A single-purpose, reproducible benchmark comparing four agent patterns over the `sw30labs/.github.wiki` corpus (pinned at commit `c130a3b` under `corpus/`):

- **RAG** (single-shot): embed query → FAISS top-k → one Claude call with retrieved chunks in context
- **Agentic-RAG**: same FAISS substrate as RAG, wrapped in a ≤8-turn tool-call loop (`retrieve`, `read_full_page`)
- **Wiki navigation**: Claude sees a precomputed sitemap up-front and uses `read_page` / `list_children` tools to navigate
- **QMD**: local hybrid retriever (BM25 FTS5 + EmbeddingGemma + Qwen3 reranker) via `qmd query --json` subprocess → one Claude call. All retrieval runs on-device; no embedding API.

Everything is a phased pipeline driven by `Makefile`. Outputs (indices, sitemaps, runs, judgments, charts) are checked in so partial reruns work.

## Commands

```bash
pip install -r requirements.txt           # openai, anthropic, faiss-cpu, tiktoken, pyyaml, numpy, matplotlib
source .env                                # sets OPENAI_API_KEY, ANTHROPIC_API_KEY

make qmd-setup                             # one-time: install Bun + @tobilu/qmd (not in `make all`)
make all                                   # full pipeline (~10 min, ~$8 of API spend)

# individual phases — each target guards its output file, so `make <phase>` is idempotent
make index     # rag/index/index_meta.json        (build FAISS from corpus/)
make sitemap   # wiki/sitemap/sitemap_meta.json   (build hierarchical sitemap.md)
make qmd-index # qmd/index_meta.json              (stage + qmd collection add + qmd embed; ~2GB GGUF on first run)
make queries   # eval/queries.jsonl               (Opus synthesizes 30 queries)
make eval      # runs/runs.jsonl                  (run all four agents over queries)
make judge     # runs/judgments.jsonl             (Opus absolute 1-5 scoring)
make report    # report/summary.json + PNGs + CSV

make clean     # wipe derived artefacts incl. qmd/staging + qmd collection (keeps eval/queries.jsonl and ~/.cache/qmd)
make qmd-nuke  # also delete ~/.cache/qmd (~2GB GGUF models). Use only to test cold-download path.
make reset     # clean + also delete the synthesized query set
```

Each Python module is also runnable standalone with positional args — see each file's `if __name__ == "__main__":` block. Useful for spot-checking a single query:

```bash
python3 rag/agent.py "What repos are in OSCAL & Compliance?" rag/index
python3 wiki/agent.py "..." corpus wiki/sitemap/sitemap.md
python3 qmd/agent.py "..."                          # uses qmd collection "wiki-vs-rag"
```

There is no test suite, linter, or type checker configured. Python 3 only; no virtualenv or lockfile.

## Architecture

The pipeline has three independent setup paths that converge at `eval/run.py`:

```
corpus/ ──► rag/chunker.py ──► rag/indexer.py ──► rag/index/              ─┐
                                                  (shared w/ rag_agentic)   │
corpus/ ──► wiki/sitemap.py ──► wiki/sitemap/sitemap.md ───────────────────┤──► eval/run.py ──► runs/runs.jsonl ──► eval/judge.py ──► runs/judgments.jsonl ──► report/aggregate.py ──► report/
                     │                                                     │
                     └──► eval/synthesize.py ──► eval/queries.jsonl ───────┤
corpus/ ──► qmd/indexer.py ──► qmd/staging/ + qmd collection + ~/.cache/qmd┘
```

### Key design rules baked into the code

- **Setup vs. per-query separation is load-bearing.** The whole experiment's thesis rests on excluding one-time costs from per-query metrics. `rag/indexer.py`, `wiki/sitemap.py`, and `qmd/indexer.py` run once and their costs are logged to `*_meta.json`. Per-query agents only measure the query-time call. Don't fold setup cost into the per-query measurement.
- **QMD retrieval is local but NOT free.** `qmd/agent.py` reports `$0.00` per-query API cost because retrieval uses on-device models. But the ~2GB GGUF model download and local compute are real infra cost — surfaced separately in `qmd/index_meta.json` (setup footprint) and the README setup-footprint table. Don't mix local compute into the per-query `cost_usd` field.
- **Sitemap summaries are not LLM-generated.** `wiki/sitemap.py` pulls the first non-header sentence (or front-matter `title`) deliberately — generating summaries with an LLM would leak LLM cost into setup. If you change this, update the README caveat.
- **All single-shot arms share the same model, temperature, `max_tokens`, and SYSTEM prompt** (`claude-sonnet-4-6`, temp 0, `max_tokens=700`) for a fair comparison. The constants and SYSTEM live at the top of each agent file. Keep them verbatim-in-sync across `rag/agent.py` and `qmd/agent.py` (they differ only in the retrieval step). Agentic arms (`wiki/agent.py`, `rag_agentic/agent.py`) share the same model constants.
- **Pricing is hardcoded** (per-1M: Sonnet $3 in / $15 out; Opus $15/$75; embeddings $0.02; QMD retrieval = $0 local) near the top of each file that computes `cost_usd`. Update all sites together if prices change.
- **Token counting uses `tiktoken` `cl100k_base`** for corpus/sitemap measurement; Anthropic's `usage.input_tokens` / `usage.output_tokens` are the ground truth for per-query cost.
- **Judge randomizes A/B ordering** (`eval/judge.py`, seeded `random.Random(7)`) and `report/aggregate.py` de-blinds via the `a_agent`/`b_agent` fields. Don't break this contract.
- **Head-to-head ties** in `report/aggregate.py` are defined as `|Δquality| < 0.17` (≈0.5 on any single axis). Baked-in threshold — change deliberately.

### RAG pipeline details

- `rag/chunker.py` splits markdown on H1/H2/H3 headers, then falls back to an 800-token sliding window with 150-token overlap. Files whose name starts with `_` (e.g. `_Sidebar.md`, `_Footer.md`) are skipped everywhere — the chunker, the sitemap, `list_children`, and `qmd/indexer.py`'s staging filter.
- `rag/indexer.py` uses `text-embedding-3-small` with batch=96, FAISS `IndexFlatIP` with L2-normalized vectors (so inner-product == cosine).
- `rag/agent.py` retrieves top-5 chunks and packs them into a single Claude call with source-path citations required.
- `rag_agentic/agent.py` shares the same FAISS index (`rag/index/`) — no separate setup cost — and wraps it in a `retrieve` + `read_full_page` tool loop capped at `MAX_TURNS=8`, `TOP_K=10`.

### Wiki pipeline details

- `wiki/agent.py` runs a tool-call loop capped at `MAX_TURNS=8`. If the cap is hit, it does a final no-tools turn asking for an answer with what's been read — this counts toward token usage and is flagged via `hit_turn_cap`.
- Tool dispatcher in `wiki/agent.py` has a path-traversal guard in `_read_page`; keep it if you refactor.
- The sitemap is injected into the system prompt on every turn, so the ~5k-token sitemap is billed once per turn, not once per query — this is the dominant cost driver and the reason prompt caching is listed in the README as a natural optimization.

### QMD pipeline details

- `qmd/indexer.py` stages `corpus/**/*.md` into `qmd/staging/` (filtered — `_`-prefixed files excluded, same rule as everywhere else), then runs `qmd collection add <staging> --name wiki-vs-rag` followed by `qmd embed`. Collection is replaced on re-run (no append behavior).
- The staging dir exists because QMD's glob/collection primitives don't directly support the `_`-prefix skip rule; staging is a derived artefact (`.gitignore`d, wiped by `make clean`).
- `qmd/agent.py` shells out to `qmd query <q> -c wiki-vs-rag -n 5 --json` per call. The JSON parser tolerates both bare-list and `{"results": [...]}` shapes, and falls back across `displayPath` / `path`, `snippet` / `context` / `body` field-name variants so the arm keeps working if QMD's output schema shifts across versions.
- QMD's cache (`~/.cache/qmd/` by default, ~2GB) is NOT wiped by `make clean` — only `make qmd-nuke` removes it. `make clean` does remove the collection (`qmd collection remove wiki-vs-rag`) so a corpus change forces a re-add on next `make qmd-index`.
- The SYSTEM prompt and all Sonnet constants in `qmd/agent.py` are a verbatim copy of `rag/agent.py`. When editing either, update both.

### Eval and reporting

- `eval/synthesize.py` calls Opus in three batches (narrow/branch/cross-branch, 10 each). Each query carries ground-truth `sources` paths used later by the judge. Queries are committed to `eval/queries.jsonl` for reproducibility; regenerating them is non-deterministic (temp 0.4).
- `eval/run.py` has a `BUDGET_USD = 10.0` circuit breaker that aborts mid-run if exceeded.
- `eval/judge.py` tolerates one JSON parse failure per query (prints and skips). `report/aggregate.py` therefore must handle missing quality scores — don't assume every `(id, agent)` pair has a `quality_avg`.
- `report/aggregate.py` writes `summary.json`, `results.csv`, and three PNGs. The PNGs are referenced from the README — regenerate them if you change chart code.

## Conventions

- Paths are stored as **relative to `corpus/`** everywhere (chunks, sitemap entries, ground truths, tool inputs, citations). Don't mix absolute paths in.
- All derived artefacts land under the module that produced them (`rag/index/`, `wiki/sitemap/`, `qmd/staging/`, `qmd/index_meta.json`, `runs/`, `report/`). `make clean` knows these locations — add new outputs to the `clean` target if you introduce more. QMD's `~/.cache/qmd/` is treated as a shared system cache, not a repo artefact, and is only wiped by `make qmd-nuke`.
- `_private/` exists but is not part of the experiment; treat as out-of-scope.
