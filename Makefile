.PHONY: all index sitemap queries eval judge report clean reset

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

eval: runs/runs.jsonl
runs/runs.jsonl: rag/index/index_meta.json wiki/sitemap/sitemap_meta.json eval/queries.jsonl
	$(PY) eval/run.py eval/queries.jsonl $(CORPUS) rag/index wiki/sitemap/sitemap.md runs/runs.jsonl

judge: runs/judgments.jsonl
runs/judgments.jsonl: runs/runs.jsonl
	$(PY) eval/judge.py runs/runs.jsonl $(CORPUS) runs/judgments.jsonl

report: report/summary.json
report/summary.json: runs/judgments.jsonl
	$(PY) report/aggregate.py runs/runs.jsonl runs/judgments.jsonl report

clean:
	rm -rf rag/index wiki/sitemap runs report/*.png report/*.csv report/summary.json

reset: clean
	rm -f eval/queries.jsonl eval/synth_meta.json
