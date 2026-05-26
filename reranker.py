"""Extension 2: Cross-encoder reranker.

Wraps the Qwen3-Reranker HTTP endpoint exposed by reranker_call.rerank and
adapts it to consume DenseRetriever Hit objects.

Pipeline position: between dense retrieval (Stage 1, recall-oriented) and
generative knowledge selection (Stage 2). Dense retrieval returns a coarse
top-N pool; the reranker scores each (query, parent_paragraph) pair with full
cross-attention and keeps the top-K with the highest reranker scores.

We rerank against a single canonical query (the rewrite). For multi-query
retrieval the variants are used to *recall* the pool; the reranker uses the
canonical question for *precision*.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

from dense_retriever import Hit
from reranker_call import DEFAULT_INSTRUCTION, rerank as _rerank


@dataclass
class RerankedHit:
    parent_idx: int
    title: str
    text: str
    best_prop: str
    dense_score: float          # cosine score from dense retriever
    rerank_score: float         # cross-encoder relevance score


def rerank_hits(
    query: str,
    hits: Sequence[Hit],
    top_k: int = 5,
    instruction: str = DEFAULT_INSTRUCTION,
) -> list[RerankedHit]:
    """Rerank parent paragraphs from a Hit list against `query`.

    Returns the top_k hits sorted by reranker score descending. The dense
    score is preserved so downstream stages can see both signals.
    """
    if not hits:
        return []
    docs = [h.text for h in hits]
    raw = _rerank(query, docs, instruction=instruction)
    out: list[RerankedHit] = []
    for r in raw:
        h = hits[r["index"]]
        out.append(RerankedHit(
            parent_idx=h.parent_idx,
            title=h.title,
            text=h.text,
            best_prop=h.best_prop,
            dense_score=h.score,
            rerank_score=float(r["score"]),
        ))
    return out[:top_k]


async def arerank_hits(
    query: str,
    hits: Sequence[Hit],
    top_k: int = 5,
    instruction: str = DEFAULT_INSTRUCTION,
) -> list[RerankedHit]:
    """Async variant. The reranker endpoint is one synchronous POST per call,
    so we offload it to a thread to avoid blocking the event loop.
    """
    return await asyncio.to_thread(rerank_hits, query, hits, top_k, instruction)


if __name__ == "__main__":
    # Smoke test — depends on dense_retriever index being available.
    from dense_retriever import DenseRetriever, index_exists

    if not index_exists():
        print("no dense index found; run `python3 demo.py` first to build one")
        raise SystemExit(1)

    r = DenseRetriever()
    r.load()
    query = "Who wrote the Foundation series of science fiction novels?"
    hits = r.retrieve([query], top_k_parents=10, top_n_props=20)
    print(f"dense top-{len(hits)}:")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] dense={h.score:.3f}  {h.title}")
    print()
    reranked = rerank_hits(query, hits, top_k=5)
    print(f"reranked top-{len(reranked)}:")
    for i, h in enumerate(reranked, 1):
        print(f"  [{i}] rerank={h.rerank_score:.4f}  dense={h.dense_score:.3f}  {h.title}")
