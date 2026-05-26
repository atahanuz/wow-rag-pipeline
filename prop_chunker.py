"""Extension 1: Propositional chunking.

LLM-driven splitter that turns a Wikipedia paragraph into a list of atomic,
self-contained propositions with pronouns resolved.

Results are cached to a JSONL file keyed by SHA-1 of (title, text). The cache
persists across runs so re-running the demo doesn't repay the LLM cost.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
from typing import Iterable

from api_clients import DEFAULT_CONCURRENCY, achat, chat, gather_bounded

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "prop_cache.jsonl")

_SYSTEM = (
    "You decompose Wikipedia paragraphs into atomic, self-contained propositions. "
    "Each proposition must: (1) state a single fact, (2) resolve every pronoun and implicit "
    "reference using the paragraph's subject (typically the title), (3) be understandable on "
    "its own without the rest of the paragraph. Output strictly valid JSON."
)


def _key(title: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(title.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


_cache_lock = threading.Lock()
_cache: dict[str, list[str]] | None = None


def _load_cache() -> dict[str, list[str]]:
    global _cache
    if _cache is not None:
        return _cache
    _cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                _cache[rec["key"]] = rec["props"]
    return _cache


def _save_entry(key: str, props: list[str]) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with _cache_lock:
        with open(CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "props": props}, ensure_ascii=False) + "\n")
        _load_cache()[key] = props


def _extract_json_array(raw: str) -> list:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON array in model output: {raw!r}")
    arr = json.loads(m.group(0))
    return [str(x).strip() for x in arr if str(x).strip()]


def _prompt(title: str, text: str) -> str:
    return (
        f"Title: {title}\n\n"
        f"Paragraph:\n{text}\n\n"
        "Return a JSON array of atomic, self-contained propositions extracted from the paragraph. "
        "Substitute the title for every pronoun/implicit subject. Output JSON only."
    )


def chunk(title: str, text: str) -> list[str]:
    key = _key(title, text)
    cache = _load_cache()
    if key in cache:
        return cache[key]
    raw = chat(_prompt(title, text), system=_SYSTEM, temperature=0.0)
    props = _extract_json_array(raw) or [text]
    _save_entry(key, props)
    return props


async def achunk(title: str, text: str) -> list[str]:
    key = _key(title, text)
    cache = _load_cache()
    if key in cache:
        return cache[key]
    raw = await achat(_prompt(title, text), system=_SYSTEM, temperature=0.0)
    props = _extract_json_array(raw) or [text]
    _save_entry(key, props)
    return props


async def achunk_many(
    items: Iterable[tuple[str, str]],
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: bool = True,
) -> list[list[str]]:
    items = list(items)
    results: list[list[str] | None] = [None] * len(items)
    todo = []
    cache = _load_cache()
    for i, (title, text) in enumerate(items):
        k = _key(title, text)
        if k in cache:
            results[i] = cache[k]
        else:
            todo.append(i)
    if progress:
        print(f"prop_chunker: {len(items) - len(todo)} cached, {len(todo)} to chunk")
    if todo:
        def _report(n: int) -> None:
            if progress and (n % 5 == 0 or n == len(todo)):
                print(f"  chunked {n}/{len(todo)}")

        awaited = await gather_bounded(
            [achunk(*items[i]) for i in todo],
            concurrency=concurrency,
            on_done=_report,
        )
        for i, props in zip(todo, awaited):
            results[i] = props
    return [r or [] for r in results]


def chunk_many(
    items: Iterable[tuple[str, str]],
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: bool = True,
) -> list[list[str]]:
    """Sync convenience wrapper that runs the async batch chunker in its own loop."""
    return asyncio.run(achunk_many(items, concurrency=concurrency, progress=progress))


if __name__ == "__main__":
    title = "Apollo 11"
    text = (
        "The Apollo 11 mission launched in 1969. It was the first spaceflight to land humans "
        "on the Moon. Neil Armstrong and Buzz Aldrin were the primary astronauts."
    )
    for p in chunk(title, text):
        print("-", p)
