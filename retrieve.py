"""Interactive BM25 retriever over the WoW passage corpus.

Usage:
    python3 retrieve.py            # interactive loop
    python3 retrieve.py "question"  # single query, prints top-5
"""
import json
import os
import re
import sys
import time

from rank_bm25 import BM25Okapi

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "data", "corpus.jsonl")
TOP_K = 5

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def load_corpus(path: str):
    titles, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            titles.append(rec["title"])
            texts.append(rec["text"])
    return titles, texts


def build_index(texts):
    t0 = time.time()
    print(f"Tokenizing {len(texts)} passages...", flush=True)
    tokenized = [tokenize(t) for t in texts]
    print(f"  done in {time.time() - t0:.1f}s. Building BM25...", flush=True)
    t1 = time.time()
    bm25 = BM25Okapi(tokenized)
    print(f"  done in {time.time() - t1:.1f}s.", flush=True)
    return bm25


def search(bm25, titles, texts, query: str, k: int = TOP_K):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(scores[i], titles[i], texts[i]) for i in top_idx]


def format_hit(rank: int, score: float, title: str, text: str, max_chars: int = 400) -> str:
    body = text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."
    return f"[{rank}] score={score:.3f}  title={title}\n    {body}"


def main():
    if not os.path.exists(CORPUS_PATH):
        print(f"corpus not found at {CORPUS_PATH}. Run build_corpus.py first.", file=sys.stderr)
        sys.exit(1)
    titles, texts = load_corpus(CORPUS_PATH)
    bm25 = build_index(texts)

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        hits = search(bm25, titles, texts, query)
        print(f"\nQuery: {query}\n")
        for i, (s, t, x) in enumerate(hits, 1):
            print(format_hit(i, s, t, x))
            print()
        return

    print("\nBM25 ready. Type a question (or 'quit' to exit).")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break
        hits = search(bm25, titles, texts, query)
        if not hits:
            print("(no results)")
            continue
        for i, (s, t, x) in enumerate(hits, 1):
            print(format_hit(i, s, t, x))
            print()


if __name__ == "__main__":
    main()
