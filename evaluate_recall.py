"""Ablation: Recall@K on WoW (test + valid random splits).

For every (apprentice_query, gold_passage_title) pair whose gold title is in
our index, we compute Recall@K under four retrieval conditions to ablate
Extension 1 (proposition-level indexing) and Extension 2 (cross-encoder
reranker):

    paragraph              : embed each parent paragraph as a single vector
    paragraph + rerank     : paragraph baseline then cross-encoder
    proposition (ours)     : decompose each paragraph into propositions,
                             retrieve at the proposition level, collapse
                             scores back to parents (highest-scoring
                             proposition wins per parent)
    proposition + rerank   : the full retrieval pipeline used by the system

All conditions share the same query rewriter output (rewrite + 2 paraphrases)
and the same dense pool depth N=10.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Iterable

import numpy as np

from api_clients import embed_batch
from dense_retriever import DenseRetriever, index_exists
from query_rewriter import rewrite as rewrite_sync
from reranker import RerankedHit, rerank_hits
from dense_retriever import Hit

DENSE_POOL = 10
KS = (1, 3, 5)
EVAL_PATHS = ["data/test_random_split.json", "data/valid_random_split.json"]


# ---------------------------------------------------------------------------
# Eval set
# ---------------------------------------------------------------------------

def load_eligible(index_titles: set[str]) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in EVAL_PATHS:
        with open(path) as f:
            data = json.load(f)
        for d in data:
            prev_apprentice: str | None = None
            for t in d["dialog"]:
                sp = t.get("speaker", "")
                text = t.get("text", "")
                if "Apprentice" in sp:
                    prev_apprentice = text
                elif "Wizard" in sp:
                    cp = t.get("checked_passage", {}) or {}
                    if cp and prev_apprentice:
                        title = list(cp.values())[0]
                        if title in index_titles and title != "no_passages_used":
                            pairs.add((prev_apprentice, title))
                    prev_apprentice = None
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Paragraph baseline: one vector per parent (no propositions)
# ---------------------------------------------------------------------------

def build_paragraph_baseline(parent_texts: list[str]) -> np.ndarray:
    print(f"embedding {len(parent_texts)} parent paragraphs for the baseline...")
    vecs = np.array(embed_batch(parent_texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vecs / norms


def paragraph_retrieve(
    para_emb: np.ndarray,
    parents: list[dict],
    query_variants: list[str],
    top_k: int,
) -> list[Hit]:
    qvecs = np.array(embed_batch(query_variants), dtype=np.float32)
    qnorms = np.linalg.norm(qvecs, axis=1, keepdims=True)
    qnorms = np.where(qnorms == 0, 1, qnorms)
    qvecs = qvecs / qnorms
    sims = qvecs @ para_emb.T            # (Q, N_parents)
    # max-pool across variants
    pooled = sims.max(axis=0)            # (N_parents,)
    order = np.argsort(-pooled)[:top_k]
    return [
        Hit(
            parent_idx=int(i),
            title=parents[int(i)]["title"],
            text=parents[int(i)]["text"],
            best_prop=parents[int(i)]["text"][:100],
            score=float(pooled[int(i)]),
        )
        for i in order
    ]


# ---------------------------------------------------------------------------
# Recall helper
# ---------------------------------------------------------------------------

def recall_at_k(ranked_titles: Iterable[str], gold: str, ks: Iterable[int]) -> dict[int, int]:
    titles = list(ranked_titles)
    return {k: int(gold in titles[:k]) for k in ks}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    assert index_exists(), "build the dense index first"

    r = DenseRetriever()
    r.load()
    index_titles = {p["title"] for p in r.parents}

    pairs = load_eligible(index_titles)
    print(f"loaded {len(pairs)} (query, gold_title) pairs")
    print("gold-title distribution:")
    for k, v in Counter(g for _, g in pairs).most_common():
        print(f"  {v:3d}  {k}")
    print()

    # Build paragraph baseline once.
    parent_texts = [p["text"] for p in r.parents]
    para_emb = build_paragraph_baseline(parent_texts)

    conditions = [
        ("paragraph",            False),
        ("paragraph+rerank",     True),
        ("proposition",          False),
        ("proposition+rerank",   True),
    ]
    correct = {c[0]: {k: 0 for k in KS} for c in conditions}
    per_query: list[dict] = []

    t_start = time.perf_counter()
    for i, (query, gold) in enumerate(pairs, 1):
        rw = rewrite_sync(query, history=(), num_paraphrases=2)
        qv = rw.all_queries

        # PARAGRAPH path -----------------------------------------------------
        para_hits = paragraph_retrieve(para_emb, r.parents, qv, top_k=DENSE_POOL)
        para_titles = [h.title for h in para_hits]
        para_rerank = rerank_hits(rw.rewrite, para_hits, top_k=DENSE_POOL) if para_hits else []
        para_rerank_titles = [h.title for h in para_rerank]

        # PROPOSITION path ---------------------------------------------------
        prop_hits = r.retrieve(qv, top_k_parents=DENSE_POOL, top_n_props=20)
        prop_titles = [h.title for h in prop_hits]
        prop_rerank = rerank_hits(rw.rewrite, prop_hits, top_k=DENSE_POOL) if prop_hits else []
        prop_rerank_titles = [h.title for h in prop_rerank]

        per_cond = {
            "paragraph":          para_titles,
            "paragraph+rerank":   para_rerank_titles,
            "proposition":        prop_titles,
            "proposition+rerank": prop_rerank_titles,
        }
        for cond, titles in per_cond.items():
            r2 = recall_at_k(titles, gold, KS)
            for k in KS:
                correct[cond][k] += r2[k]

        per_query.append({
            "query": query, "rewrite": rw.rewrite, "gold": gold,
            "top5": {c: per_cond[c][:5] for c in per_cond},
        })

        marks = " ".join(
            f"{c.split('+')[0][0]}{('R' if '+rerank' in c else '')}={('Y' if gold in per_cond[c][:5] else 'N')}"
            for c in per_cond
        )
        print(f"[{i:2d}/{len(pairs)}] gold={gold[:28]:28.28s}  {marks}  rwr={rw.rewrite[:40]!r}")

    elapsed = time.perf_counter() - t_start
    n = len(pairs)
    print()
    print(f"elapsed: {elapsed:.1f}s for {n} queries  ({elapsed / max(n, 1):.1f}s/query)")
    print()
    print(f"{'condition':<22}  " + "  ".join(f"R@{k}".rjust(7) for k in KS))
    print("-" * (22 + 2 + 9 * len(KS)))
    rows = []
    for cond, _ in conditions:
        cells = "  ".join(f"{correct[cond][k]/n:6.3f}".rjust(7) for k in KS)
        print(f"{cond:<22}  {cells}")
        rows.append({
            "condition": cond,
            **{f"R@{k}": correct[cond][k] / n for k in KS},
            **{f"hits@{k}": correct[cond][k] for k in KS},
        })

    out = {
        "config": {"dense_pool": DENSE_POOL, "ks": list(KS), "splits": EVAL_PATHS},
        "n_pairs": n,
        "elapsed_seconds": round(elapsed, 1),
        "rows": rows,
        "per_query": per_query,
    }
    with open("eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("wrote eval_results.json")


if __name__ == "__main__":
    main()
