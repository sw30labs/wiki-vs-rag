.PHONY: all index sitemap queries eval judge report clean reset qmd-setup qmd-index qmd-nuke

PY := python3
CORPUS := corpus

all: report/summary.json

index: rag/index/index_meta.json
rag/index/index_meta.json:
	$(PY) rag/indexer.py $(CORPUS) rag/index

sitemap: wiki/sitemap/sitemap_meta.json
wiki/sitemap/sitemap_meta.json:
	$(PY) wiki/sitemap.py $(CORPUS) wiki/sitemap

queries: eval/queries.jsonl
eval/queries.jsonl: wiki/sitemap/sitemap_meta.json
	$(PY) eval/synthesize.py $(CORPUS) wiki/sitemap/sitemap.md eval/queries.jsonl

# QMD is the local-hybrid retrieval arm. Bootstrap is one-time and deliberately
# NOT wired into `make all` — it installs a global npm package and downloads
# ~2 GB of GGUF models on first use.
qmd-setup:
	@which bun >/dev/null 2>&1 || { echo "Install Bun first: https://bun.sh"; exit 1; }
	@which qmd >/dev/null 2>&1 || npm install -g @tobilu/qmd
	@echo "QMD ready. Run \`make qmd-index\` to build the collection (downloads ~2 GB models on first run)."

qmd-index: qmd/index_meta.json
qmd/index_meta.json:
	$(PY) qmd/indexer.py $(CORPUS) qmd

eval: runs/runs.jsonl
runs/runs.jsonl: rag/index/index_meta.json wiki/sitemap/sitemap_meta.json qmd/index_meta.json eval/queries.jsonl
	$(PY) eval/run.py eval/queries.jsonl $(CORPUS) rag/index wiki/sitemap/sitemap.md runs/runs.jsonl

judge: runs/judgments.jsonl
runs/judgments.jsonl: runs/runs.jsonl
	$(PY) eval/judge.py runs/runs.jsonl $(CORPUS) runs/judgments.jsonl

report: report/summary.json
report/summary.json: runs/judgments.jsonl
	$(PY) report/aggregate.py runs/runs.jsonl runs/judgments.jsonl report

clean:
	rm -rf rag/index wiki/sitemap runs report/*.png report/*.csv report/summary.json
	rm -rf qmd/index_meta.json qmd/staging
	@which qmd >/dev/null 2>&1 && qmd collection remove wiki-vs-rag 2>/dev/null || true

# Wipe QMD's local model + sqlite cache (~2 GB). Use only to test the cold-download path.
qmd-nuke:
	rm -rf ~/.cache/qmd

reset: clean
	rm -f eval/queries.jsonl eval/synth_meta.json
