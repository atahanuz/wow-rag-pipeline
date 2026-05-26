"""Stage 1 + Extension 1: dense retrieval over proposition-indexed passages.

Build:
  - Take a list of (title, paragraph) records.
  - Chunk each paragraph into propositions (Extension 1, via prop_chunker).
  - Embed each proposition (qwen3-embedding-8b).
  - Save embeddings + metadata to data/dense_index/.

Query:
  - Embed the query (or several variants).
  - Cosine top-K over propositions.
  - Collapse hits back to their parent paragraphs (Extension 1), keeping the
    best-scoring proposition per parent.

The index is small (numpy + jsonl). No vector DB needed at this scale.
For full WoW corpus we'll swap in Chroma; this is the thin slice.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from api_clients import DEFAULT_CONCURRENCY, EMBED_DIM, aembed, aembed_batch, embed_batch, gather_bounded
from prop_chunker import achunk_many

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "dense_index")
EMB_PATH = os.path.join(INDEX_DIR, "props.npy")
META_PATH = os.path.join(INDEX_DIR, "props.jsonl")
PARENTS_PATH = os.path.join(INDEX_DIR, "parents.jsonl")


@dataclass
class Hit:
    parent_idx: int
    title: str
    text: str
    best_prop: str
    score: float


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms


async def _aembed_concurrent(texts: Sequence[str], concurrency: int = DEFAULT_CONCURRENCY, progress: bool = True) -> np.ndarray:
    def _report(n: int) -> None:
        if progress and (n % 25 == 0 or n == len(texts)):
            print(f"  embedded {n}/{len(texts)}")

    vectors = await gather_bounded(
        [aembed(t) for t in texts],
        concurrency=concurrency,
        on_done=_report,
    )
    return np.array(vectors, dtype=np.float32)


class DenseRetriever:
    def __init__(self):
        self.parents: list[dict] = []          # [{title, text}]
        self.prop_texts: list[str] = []        # one row per proposition
        self.prop_parent: list[int] = []       # parent_idx for each prop
        self.embeddings: np.ndarray | None = None  # (N, D) L2-normalized

    # ---------- build ----------

    async def abuild(self, parents: Sequence[tuple[str, str]], concurrency: int = DEFAULT_CONCURRENCY) -> None:
        print(f"DenseRetriever.build: {len(parents)} parent paragraphs")
        prop_lists = await achunk_many(parents, concurrency=concurrency)
        self.parents = [{"title": t, "text": x} for (t, x) in parents]
        self.prop_texts = []
        self.prop_parent = []
        for parent_idx, props in enumerate(prop_lists):
            for p in props:
                self.prop_texts.append(p)
                self.prop_parent.append(parent_idx)
        print(f"DenseRetriever.build: {len(self.prop_texts)} propositions; embedding...")
        embs = await _aembed_concurrent(self.prop_texts, concurrency=concurrency)
        assert embs.shape == (len(self.prop_texts), EMBED_DIM), embs.shape
        self.embeddings = _normalize(embs)

    def build(self, parents: Sequence[tuple[str, str]], concurrency: int = DEFAULT_CONCURRENCY) -> None:
        asyncio.run(self.abuild(parents, concurrency=concurrency))

    def save(self, index_dir: str = INDEX_DIR) -> None:
        if self.embeddings is None:
            raise RuntimeError("call build() before save()")
        os.makedirs(index_dir, exist_ok=True)
        np.save(os.path.join(index_dir, "props.npy"), self.embeddings)
        with open(os.path.join(index_dir, "props.jsonl"), "w", encoding="utf-8") as f:
            for t, p in zip(self.prop_texts, self.prop_parent):
                f.write(json.dumps({"text": t, "parent": p}, ensure_ascii=False) + "\n")
        with open(os.path.join(index_dir, "parents.jsonl"), "w", encoding="utf-8") as f:
            for rec in self.parents:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"DenseRetriever.save: wrote index to {index_dir}")

    def load(self, index_dir: str = INDEX_DIR) -> None:
        self.embeddings = np.load(os.path.join(index_dir, "props.npy"))
        self.prop_texts, self.prop_parent = [], []
        with open(os.path.join(index_dir, "props.jsonl"), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.prop_texts.append(rec["text"])
                self.prop_parent.append(rec["parent"])
        self.parents = []
        with open(os.path.join(index_dir, "parents.jsonl"), encoding="utf-8") as f:
            for line in f:
                self.parents.append(json.loads(line))

    # ---------- query ----------

    def _search_prop(self, query_vecs: np.ndarray, top_n_props: int) -> np.ndarray:
        # query_vecs: (Q, D) normalized
        # returns: (Q, top_n_props) prop indices, scores
        sims = query_vecs @ self.embeddings.T  # (Q, N)
        idx = np.argpartition(-sims, kth=min(top_n_props, sims.shape[1] - 1), axis=1)[:, :top_n_props]
        # sort each row by actual score
        rows = np.arange(sims.shape[0])[:, None]
        ordered = np.argsort(-sims[rows, idx], axis=1)
        idx = idx[rows, ordered]
        scores = sims[rows, idx]
        return idx, scores

    async def aretrieve(self, queries: Sequence[str], top_k_parents: int = 5, top_n_props: int = 50) -> list[Hit]:
        if self.embeddings is None:
            raise RuntimeError("index not loaded/built")
        if not queries:
            return []
        qvecs_raw = await aembed_batch(queries)
        return self._search_and_collapse(np.array(qvecs_raw, dtype=np.float32), top_k_parents, top_n_props)

    def retrieve(self, queries: Sequence[str], top_k_parents: int = 5, top_n_props: int = 50) -> list[Hit]:
        """Embed each query, search propositions, collapse to unique parent paragraphs.

        If multiple queries are given, scores for the same proposition are max-pooled.
        Within a parent paragraph, the best-matching proposition wins.
        """
        if self.embeddings is None:
            raise RuntimeError("index not loaded/built")
        if not queries:
            return []
        qvecs_raw = embed_batch(queries)
        return self._search_and_collapse(np.array(qvecs_raw, dtype=np.float32), top_k_parents, top_n_props)

    def _search_and_collapse(self, qvecs_raw: np.ndarray, top_k_parents: int, top_n_props: int) -> list[Hit]:
        qvecs = _normalize(qvecs_raw)
        idx, scores = self._search_prop(qvecs, top_n_props=top_n_props)

        # max-pool proposition scores across queries
        best_per_prop: dict[int, float] = {}
        for q in range(idx.shape[0]):
            for j in range(idx.shape[1]):
                pid = int(idx[q, j])
                s = float(scores[q, j])
                if s > best_per_prop.get(pid, -1e9):
                    best_per_prop[pid] = s

        # collapse to parent paragraphs
        best_per_parent: dict[int, tuple[float, int]] = {}
        for pid, s in best_per_prop.items():
            par = self.prop_parent[pid]
            if s > best_per_parent.get(par, (-1e9, -1))[0]:
                best_per_parent[par] = (s, pid)

        ranked = sorted(best_per_parent.items(), key=lambda x: -x[1][0])[:top_k_parents]
        hits = []
        for par_idx, (score, prop_id) in ranked:
            parent = self.parents[par_idx]
            hits.append(Hit(
                parent_idx=par_idx,
                title=parent["title"],
                text=parent["text"],
                best_prop=self.prop_texts[prop_id],
                score=score,
            ))
        return hits


def index_exists(index_dir: str = INDEX_DIR) -> bool:
    return all(os.path.exists(os.path.join(index_dir, f)) for f in ("props.npy", "props.jsonl", "parents.jsonl"))
