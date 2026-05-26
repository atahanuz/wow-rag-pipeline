"""Extension 3: Per-dialogue passage cache.

Conversations tend to stay on topic, so the passage that GenKS chose for turn
N often answers turn N+1 as well. This module stores that passage (with its
embedding, turn number, and Q² score) per dialogue and lets the server skip
dense retrieval + reranker + GenKS when a cached passage already matches the
new query above a threshold.

Per the proposal:
    - LRU bounded to N=10 entries per dialogue.
    - Cosine similarity against the query embedding.
    - On hit, the cached passage goes straight to Stage 3 (KEDiT).
    - FLARE-triggered re-retrieval also consults the cache first.

This is an engineering optimization (the proposal: "The cache system doesn't
improve the accuracy of the system but fastens the inference by removing
redundant calculations.").
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from reranker import RerankedHit

MAX_ENTRIES_PER_DIALOGUE = 10


@dataclass
class CachedEntry:
    hit: RerankedHit                # the passage (title, text, scores)
    embedding: np.ndarray           # L2-normalized parent-text embedding (1-D float32)
    turn: int                       # 1-based turn number when this passage was added
    q2_score: Optional[float] = None


@dataclass
class CacheLookupResult:
    hit: bool
    similarity: float               # best similarity found (whether hit or not)
    entry: Optional[CachedEntry] = None


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return (v / n).astype(np.float32)


class DialogueCache:
    """In-memory per-dialogue LRU cache. Process-local; cleared on restart."""

    def __init__(self, max_per_dialogue: int = MAX_ENTRIES_PER_DIALOGUE):
        self.max = max_per_dialogue
        # OrderedDict[entry_id -> CachedEntry] per dialogue; LRU via move_to_end
        self._cache: dict[str, "OrderedDict[int, CachedEntry]"] = {}
        self._next_key: dict[str, int] = {}

    # --- ids ---------------------------------------------------------------

    def new_dialogue(self) -> str:
        d = str(uuid.uuid4())
        self._cache[d] = OrderedDict()
        self._next_key[d] = 0
        return d

    def ensure(self, dialogue_id: str) -> None:
        if dialogue_id not in self._cache:
            self._cache[dialogue_id] = OrderedDict()
            self._next_key[dialogue_id] = 0

    def reset(self, dialogue_id: str) -> None:
        self._cache.pop(dialogue_id, None)
        self._next_key.pop(dialogue_id, None)

    def size(self, dialogue_id: str) -> int:
        return len(self._cache.get(dialogue_id, {}))

    # --- core ops ---------------------------------------------------------

    def lookup(
        self,
        dialogue_id: str,
        query_embedding: np.ndarray,
        tau_cache: float,
    ) -> CacheLookupResult:
        bucket = self._cache.get(dialogue_id)
        if not bucket:
            return CacheLookupResult(hit=False, similarity=0.0)
        q = _normalize(np.asarray(query_embedding, dtype=np.float32))
        best_key: int | None = None
        best_sim = -1.0
        for key, entry in bucket.items():
            sim = float(np.dot(q, entry.embedding))
            if sim > best_sim:
                best_sim = sim
                best_key = key
        if best_key is not None and best_sim >= tau_cache:
            # mark this entry as most-recently-used
            bucket.move_to_end(best_key)
            return CacheLookupResult(hit=True, similarity=best_sim, entry=bucket[best_key])
        return CacheLookupResult(hit=False, similarity=max(best_sim, 0.0))

    def add(
        self,
        dialogue_id: str,
        hit: RerankedHit,
        embedding: np.ndarray,
        turn: int,
        q2_score: Optional[float] = None,
    ) -> None:
        self.ensure(dialogue_id)
        # If the same parent_idx is already cached, refresh it (move-to-end + update q2).
        for key, entry in self._cache[dialogue_id].items():
            if entry.hit.parent_idx == hit.parent_idx:
                entry.q2_score = q2_score if q2_score is not None else entry.q2_score
                entry.turn = turn
                self._cache[dialogue_id].move_to_end(key)
                return
        # Insert new entry.
        self._next_key[dialogue_id] += 1
        new_key = self._next_key[dialogue_id]
        self._cache[dialogue_id][new_key] = CachedEntry(
            hit=hit,
            embedding=_normalize(np.asarray(embedding, dtype=np.float32)),
            turn=turn,
            q2_score=q2_score,
        )
        # LRU eviction.
        while len(self._cache[dialogue_id]) > self.max:
            self._cache[dialogue_id].popitem(last=False)

    def update_q2(self, dialogue_id: str, parent_idx: int, q2_score: float) -> None:
        bucket = self._cache.get(dialogue_id)
        if not bucket:
            return
        for entry in bucket.values():
            if entry.hit.parent_idx == parent_idx:
                entry.q2_score = q2_score
                return


# Process-wide singleton.
_global_cache: DialogueCache | None = None


def get_global_cache() -> DialogueCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = DialogueCache()
    return _global_cache
