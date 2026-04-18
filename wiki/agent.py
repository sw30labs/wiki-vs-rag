"""Wiki-fed navigation agent.

Claude gets the sitemap up-front plus two tools:
  - read_page(path)        returns the full markdown of a page
  - list_children(prefix)  returns pages under prefix/
Tool-call loop with a hard max-turn cap. Every turn's usage is summed.
"""
from __future__ import annotations
import os, json, time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

CHAT_MODEL = "claude-sonnet-4-6"
MAX_OUT = 700
MAX_TURNS = 8
SONNET_IN = 3.0
SONNET_OUT = 15.0

SYSTEM_TEMPLATE = """You are a helpful assistant answering questions about the sw30labs organization wiki.

You are given a SITEMAP below — a compact tree of the wiki showing every page path, title, and a one-sentence summary. Use it to decide which page(s) to open.

Strategy:
1. Scan the sitemap and pick the most relevant page(s) for the question.
2. Use `read_page(path)` to load the content of one page at a time. Start narrow; expand only if needed.
3. Use `list_children(prefix)` only if the sitemap entries are insufficient.
4. STOP reading pages as soon as you have enough to answer. Do not load unnecessary pages.
5. Answer concisely (2-6 sentences unless the question needs more). Cite source paths inline in square brackets like [path/to/file.md].

If the wiki genuinely doesn't cover the question, say so plainly — don't guess.

SITEMAP:

{sitemap}
"""

TOOLS = [
    {
        "name": "read_page",
        "description": "Read the full markdown content of a single wiki page.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from the corpus root, e.g. 'wiki/ai-security/ai-security.md'"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_children",
        "description": "List all pages whose path begins with the given prefix. Use to explore a subtree.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prefix": {"type": "string", "description": "Path prefix, e.g. 'wiki/articles/'"}
            },
            "required": ["prefix"],
        },
    },
]


@dataclass
class WikiResult:
    query: str
    answer: str
    pages_read: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    turns: int = 0
    hit_turn_cap: bool = False


def _read_page(corpus_root: Path, path: str) -> str:
    p = (corpus_root / path).resolve()
    # guard against path traversal
    if corpus_root.resolve() not in p.parents and p != corpus_root.resolve():
        return f"error: path outside corpus ({path})"
    if not p.exists() or not p.is_file():
        return f"error: not found ({path})"
    return p.read_text(encoding="utf-8", errors="ignore")


def _list_children(corpus_root: Path, prefix: str) -> str:
    hits = []
    for md in sorted(corpus_root.rglob("*.md")):
        if md.name.startswith("_"):
            continue
        rel = str(md.relative_to(corpus_root))
        if rel.startswith(prefix):
            hits.append(rel)
    if not hits:
        return f"(no pages under prefix {prefix!r})"
    return "\n".join(hits)


def answer(query: str, corpus_root: Path, sitemap_md: str) -> WikiResult:
    ac = anthropic.Anthropic()
    sys_prompt = SYSTEM_TEMPLATE.format(sitemap=sitemap_md)
    messages: list[dict] = [{"role": "user", "content": query}]

    res = WikiResult(query=query, answer="")
    t0 = time.perf_counter()

    for turn in range(MAX_TURNS):
        res.turns = turn + 1
        r = ac.messages.create(
            model=CHAT_MODEL,
            max_tokens=MAX_OUT,
            system=sys_prompt,
            temperature=0,
            tools=TOOLS,
            messages=messages,
        )
        res.input_tokens += r.usage.input_tokens
        res.output_tokens += r.usage.output_tokens

        if r.stop_reason == "end_turn":
            # extract text
            for b in r.content:
                if b.type == "text":
                    res.answer += b.text
            break

        if r.stop_reason == "tool_use":
            # record assistant turn
            messages.append({"role": "assistant", "content": r.content})
            # handle all tool uses this turn
            tool_results = []
            for b in r.content:
                if b.type != "tool_use":
                    continue
                if b.name == "read_page":
                    path = b.input.get("path", "")
                    res.pages_read.append(path)
                    out = _read_page(corpus_root, path)
                elif b.name == "list_children":
                    out = _list_children(corpus_root, b.input.get("prefix", ""))
                else:
                    out = f"error: unknown tool {b.name}"
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": out,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # any other stop reason — bail out
        for b in r.content:
            if b.type == "text":
                res.answer += b.text
        break
    else:
        res.hit_turn_cap = True
        # try one final turn asking for the answer with no tools
        r = ac.messages.create(
            model=CHAT_MODEL,
            max_tokens=MAX_OUT,
            system=sys_prompt + "\n\nYou have reached the page-reading limit. Answer now with what you have.",
            temperature=0,
            messages=messages + [{"role": "user", "content": "Give your final answer now."}],
        )
        res.input_tokens += r.usage.input_tokens
        res.output_tokens += r.usage.output_tokens
        for b in r.content:
            if b.type == "text":
                res.answer += b.text

    res.latency_s = round(time.perf_counter() - t0, 2)
    res.cost_usd = round(
        res.input_tokens / 1_000_000 * SONNET_IN + res.output_tokens / 1_000_000 * SONNET_OUT,
        6,
    )
    return res


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What repos are in the OSCAL & Compliance category?"
    corpus = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("corpus")
    sitemap = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("wiki/sitemap/sitemap.md")
    res = answer(q, corpus, sitemap.read_text())
    print(json.dumps(res.__dict__, indent=2))
