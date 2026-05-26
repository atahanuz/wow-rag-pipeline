"""FastAPI backend for the Node UI.

Wraps the existing pipeline modules. Sync handlers — the underlying functions
already do their own concurrency where useful (asyncio.gather inside the
chunker, batched embeddings in the retriever, threaded reranker call).

Run:
    uvicorn server:app --reload --port 8000
"""
from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api_clients import embed as embed_text
from dense_retriever import DenseRetriever, INDEX_DIR, index_exists
from dialogue_cache import MAX_ENTRIES_PER_DIALOGUE, get_global_cache
from faithfulness import evaluate as faithfulness_evaluate
from flare import check as flare_check
from generator import generate as generate_answer
from genks import select as genks_select
from kedit import distill as kedit_distill
from query_rewriter import rewrite as rewrite_sync
from reranker import RerankedHit, rerank_hits

app = FastAPI(title="WoW RAG Pipeline")


# ---------------------------------------------------------------------------
# Singleton retriever (loaded on first request)
# ---------------------------------------------------------------------------

_retriever: DenseRetriever | None = None


def get_retriever() -> DenseRetriever:
    global _retriever
    if _retriever is not None:
        return _retriever
    if not index_exists():
        raise HTTPException(
            status_code=503,
            detail=f"No dense index at {INDEX_DIR}. Build it first.",
        )
    r = DenseRetriever()
    r.load()
    _retriever = r
    return r


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    role: str
    text: str


class Settings(BaseModel):
    dense_pool: int = Field(10, ge=1, le=50)
    top_n_props: int = Field(20, ge=1, le=100)
    rerank_k: int = Field(3, ge=1, le=10)
    n_paraphrases: int = Field(2, ge=0, le=4)
    gen_temperature: float = Field(0.2, ge=0.0, le=1.0)
    tau_cache: float = Field(0.70, ge=0.0, le=1.0)


class RunRequest(BaseModel):
    user_turn: str
    history: list[Turn] = []
    settings: Settings = Field(default_factory=Settings)
    dialogue_id: str | None = None     # Ext 3: per-dialogue cache key


class DenseHitOut(BaseModel):
    rank: int
    title: str
    text: str
    best_prop: str
    dense_score: float


class RerankedHitOut(DenseHitOut):
    rerank_score: float


class RewriteOut(BaseModel):
    rewrite: str
    paraphrases: list[str]


class SelectionOut(BaseModel):
    chosen_idx: int                # 0-based index into the reranked list
    chosen_label: str              # "k1", "k2", ...
    chosen_title: str
    reason: str
    prompt: str
    fallback: bool


class KeditOut(BaseModel):
    source_title: str
    source_text: str               # the original passage chosen by GenKS
    summary: str
    facts: list[str]
    brief: str                     # rendered brief that is actually fed to the generator
    prompt: str
    fallback: bool


class FlareOut(BaseModel):
    triggered: bool
    grounded: bool                  # was the draft fully grounded?
    unsupported_claims: list[str]
    retrieval_query: str
    re_retrieved: list[RerankedHitOut]   # new top hits used for refinement (may be empty)
    draft_answer: str               # the answer BEFORE refinement
    refined: bool                   # True if we successfully ran a refinement pass
    prompt: str
    fallback: bool


class QAPairOut(BaseModel):
    question: str
    response_answer: str
    knowledge_answer: str | None
    match: bool


class FaithfulnessOut(BaseModel):
    # Q²
    q2_score: float
    q2_threshold: float
    qa_pairs: list[QAPairOut]
    q2_fallback: bool
    # BEGIN
    begin_label: str                # "Fully Attributable" | "Not Attributable" | "Generic"
    begin_rationale: str
    begin_fallback: bool
    # gate
    gate_failed: bool
    # regeneration outcome (when gate failed and a runner-up existed)
    regenerated: bool
    runner_up_title: str | None
    pre_gate_answer: str | None     # the answer BEFORE regeneration (None if gate passed)


class CacheOut(BaseModel):
    dialogue_id: str
    hit: bool                       # True if main cache lookup served the passage
    similarity: float               # best cosine similarity observed (0 if cache empty)
    tau_cache: float
    cached_title: str | None        # title of the served entry (None on miss)
    cached_turn: int | None         # which turn the entry was first added on
    size_after: int                 # cache size after this turn
    max_size: int
    flare_cache_hit: bool           # True if FLARE re-retrieval also hit the cache


class RunResponse(BaseModel):
    rewrite: RewriteOut
    dense: list[DenseHitOut]
    reranked: list[RerankedHitOut]
    selection: SelectionOut | None
    kedit: KeditOut | None
    flare: FlareOut | None
    faithfulness: FaithfulnessOut | None
    cache: CacheOut
    dialogue_id: str
    answer: str
    used_titles: list[str]
    prompt: str
    timings_ms: dict[str, float]


class InfoResponse(BaseModel):
    parents: int
    propositions: int
    index_dir: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/info", response_model=InfoResponse)
def info() -> Any:
    r = get_retriever()
    return InfoResponse(
        parents=len(r.parents),
        propositions=len(r.prop_texts),
        index_dir=INDEX_DIR,
    )


@app.post("/api/dialogue/reset")
def reset_dialogue(dialogue_id: str | None = None) -> dict:
    """Clear a dialogue's passage cache. Used by the UI's 'New conversation' button."""
    cache = get_global_cache()
    if dialogue_id:
        cache.reset(dialogue_id)
    new_id = cache.new_dialogue()
    return {"dialogue_id": new_id}


@app.post("/api/run", response_model=RunResponse)
def run(req: RunRequest) -> Any:
    import time
    if not req.user_turn.strip():
        raise HTTPException(status_code=400, detail="user_turn is empty")
    r = get_retriever()
    cache = get_global_cache()
    history = [(t.role, t.text) for t in req.history]
    timings: dict[str, float] = {}

    # Resolve or create dialogue id.
    dialogue_id = req.dialogue_id or cache.new_dialogue()
    cache.ensure(dialogue_id)

    t0 = time.perf_counter()
    rw = rewrite_sync(req.user_turn, history=history, num_paraphrases=req.settings.n_paraphrases)
    timings["rewrite"] = (time.perf_counter() - t0) * 1000

    # Ext 3 — Per-dialogue cache lookup. We embed the canonical rewrite and
    # score it against every entry in this dialogue's LRU. On hit, we skip
    # dense retrieval + reranker + GenKS and feed the cached passage straight
    # to Stage 3 (KEDiT). On miss, the full pipeline runs and the GenKS-chosen
    # passage is added to the cache at the end of this turn.
    t0 = time.perf_counter()
    query_emb = np.asarray(embed_text(rw.rewrite), dtype=np.float32)
    timings["cache_embed"] = (time.perf_counter() - t0) * 1000

    cache_lookup = cache.lookup(dialogue_id, query_emb, req.settings.tau_cache)

    # These three are filled either from the cache (hit) or from the live
    # dense -> rerank -> GenKS path (miss).
    hits: list = []                       # dense top-N (empty on cache hit)
    reranked: list[RerankedHit] = []      # reranker top-K (empty on cache hit)
    selection_out: SelectionOut | None = None
    kedit_out: KeditOut | None = None
    passages_for_gen: list = []
    chosen_hit: RerankedHit | None = None  # what KEDiT distills

    if cache_lookup.hit and cache_lookup.entry is not None:
        # Cache hit: bypass dense + rerank + GenKS. Cached passage goes
        # straight into KEDiT.
        chosen_hit = cache_lookup.entry.hit
    else:
        # Cache miss: full retrieval pipeline.
        t0 = time.perf_counter()
        hits = r.retrieve(
            rw.all_queries,
            top_k_parents=req.settings.dense_pool,
            top_n_props=req.settings.top_n_props,
        )
        timings["dense"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reranked = rerank_hits(rw.rewrite, hits, top_k=req.settings.rerank_k)
        timings["rerank"] = (time.perf_counter() - t0) * 1000

        if reranked:
            t0 = time.perf_counter()
            sel = genks_select(req.user_turn, history, reranked)
            timings["genks"] = (time.perf_counter() - t0) * 1000
            selection_out = SelectionOut(
                chosen_idx=sel.chosen_idx,
                chosen_label=sel.chosen_label,
                chosen_title=sel.chosen.title,
                reason=sel.reason,
                prompt=sel.prompt,
                fallback=sel.fallback,
            )
            chosen_hit = sel.chosen

    # KEDiT runs whether we got the passage from the cache or from GenKS.
    if chosen_hit is not None:
        t0 = time.perf_counter()
        dist = kedit_distill(req.user_turn, history, chosen_hit)
        timings["kedit"] = (time.perf_counter() - t0) * 1000
        kedit_out = KeditOut(
            source_title=dist.source.title,
            source_text=dist.source.text,
            summary=dist.summary,
            facts=dist.facts,
            brief=dist.as_brief(),
            prompt=dist.prompt,
            fallback=dist.fallback,
        )
        passages_for_gen = [dist.to_passage()]

    t0 = time.perf_counter()
    gen = generate_answer(req.user_turn, history, passages_for_gen, temperature=req.settings.gen_temperature)
    timings["generate"] = (time.perf_counter() - t0) * 1000

    # Stage 4 — FLARE: grounding check on the draft. If any claim isn't supported
    # by the source knowledge, re-retrieve using the unsupported claims as the
    # query and regenerate once with the merged context.
    flare_out: FlareOut | None = None
    flare_cache_hit = False
    final_answer = gen.answer
    final_used_titles = [h.title for h in gen.used_passages]
    final_prompt = gen.prompt

    if kedit_out is not None:
        t0 = time.perf_counter()
        fc = flare_check(req.user_turn, gen.answer, kedit_out.source_text)
        timings["flare_check"] = (time.perf_counter() - t0) * 1000

        triggered = (not fc.grounded) and bool(fc.retrieval_query)
        re_retrieved_out: list[RerankedHitOut] = []
        refined = False

        if triggered:
            # Ext 3: when FLARE re-retrieves, check the cache first.
            t0 = time.perf_counter()
            flare_q_emb = np.asarray(embed_text(fc.retrieval_query), dtype=np.float32)
            flare_cache_result = cache.lookup(dialogue_id, flare_q_emb, req.settings.tau_cache)
            timings["flare_cache_lookup"] = (time.perf_counter() - t0) * 1000

            if flare_cache_result.hit and flare_cache_result.entry is not None:
                # Reuse the cached passage as the "new" top-1.
                new_reranked = [flare_cache_result.entry.hit]
                flare_cache_hit = True
            else:
                # one refinement pass: dense -> rerank -> generate, merging with the
                # current distilled passage so we don't lose what GenKS already chose.
                t0 = time.perf_counter()
                new_hits = r.retrieve(
                    [fc.retrieval_query],
                    top_k_parents=req.settings.dense_pool,
                    top_n_props=req.settings.top_n_props,
                )
                new_reranked = rerank_hits(fc.retrieval_query, new_hits, top_k=2) if new_hits else []
                timings["flare_retrieve"] = (time.perf_counter() - t0) * 1000

            re_retrieved_out = [
                RerankedHitOut(
                    rank=i + 1,
                    title=h.title,
                    text=h.text,
                    best_prop=h.best_prop,
                    dense_score=float(h.dense_score),
                    rerank_score=float(h.rerank_score),
                )
                for i, h in enumerate(new_reranked)
            ]

            if new_reranked:
                # merge original distilled passage with the best new passage
                merged_passages = [passages_for_gen[0], new_reranked[0]]
                t0 = time.perf_counter()
                regen = generate_answer(
                    req.user_turn, history, merged_passages,
                    temperature=req.settings.gen_temperature,
                )
                timings["flare_regenerate"] = (time.perf_counter() - t0) * 1000
                final_answer = regen.answer
                final_used_titles = [h.title for h in regen.used_passages]
                final_prompt = regen.prompt
                refined = True

        flare_out = FlareOut(
            triggered=triggered,
            grounded=fc.grounded,
            unsupported_claims=fc.unsupported_claims,
            retrieval_query=fc.retrieval_query,
            re_retrieved=re_retrieved_out,
            draft_answer=gen.answer,
            refined=refined,
            prompt=fc.prompt,
            fallback=fc.fallback,
        )

    # Stage 5 — Q² + BEGIN faithfulness gate. Runs Q² (question/answer
    # consistency between response and knowledge) and BEGIN (attribution
    # classification) in parallel against the post-FLARE answer. On failure,
    # regenerate ONCE using the GenKS runner-up passage (per the proposal).
    faithfulness_out: FaithfulnessOut | None = None
    if kedit_out is not None:
        knowledge_text = kedit_out.source_text
        t0 = time.perf_counter()
        fres = faithfulness_evaluate(final_answer, knowledge_text)
        timings["faithfulness"] = (time.perf_counter() - t0) * 1000

        regenerated = False
        runner_up_title: str | None = None
        pre_gate_answer: str | None = None

        if fres.failed and reranked and selection_out is not None:
            # Pick the first reranked candidate other than the one GenKS chose.
            runner_up = next(
                (h for i, h in enumerate(reranked) if i != selection_out.chosen_idx),
                None,
            )
            if runner_up is not None:
                t0 = time.perf_counter()
                runner_dist = kedit_distill(req.user_turn, history, runner_up)
                regen2 = generate_answer(
                    req.user_turn, history, [runner_dist.to_passage()],
                    temperature=req.settings.gen_temperature,
                )
                timings["faithfulness_regenerate"] = (time.perf_counter() - t0) * 1000
                pre_gate_answer = final_answer
                final_answer = regen2.answer
                final_used_titles = [h.title for h in regen2.used_passages]
                final_prompt = regen2.prompt
                runner_up_title = runner_up.title
                regenerated = True

        faithfulness_out = FaithfulnessOut(
            q2_score=fres.q2.score,
            q2_threshold=fres.q2_threshold,
            qa_pairs=[
                QAPairOut(
                    question=p.question,
                    response_answer=p.response_answer,
                    knowledge_answer=p.knowledge_answer,
                    match=p.match,
                )
                for p in fres.q2.qa_pairs
            ],
            q2_fallback=fres.q2.fallback,
            begin_label=fres.begin.label,
            begin_rationale=fres.begin.rationale,
            begin_fallback=fres.begin.fallback,
            gate_failed=fres.failed,
            regenerated=regenerated,
            runner_up_title=runner_up_title,
            pre_gate_answer=pre_gate_answer,
        )

    # Ext 3 — cache population.
    # - On miss with a successful chosen_hit: embed the parent passage text and
    #   insert into the dialogue's LRU.
    # - On hit: refresh the existing entry's Q² score (and move-to-end was already
    #   done by lookup).
    turn_number = len(history) // 2 + 1
    final_q2: float | None = (
        faithfulness_out.q2_score if faithfulness_out is not None else None
    )
    if cache_lookup.hit and chosen_hit is not None and final_q2 is not None:
        cache.update_q2(dialogue_id, chosen_hit.parent_idx, final_q2)
    elif (not cache_lookup.hit) and chosen_hit is not None:
        # Embed the best proposition (focused single-fact sentence) instead of
        # the full parent paragraph: short, query-shaped text yields much higher
        # cosine similarity against follow-up queries on the same topic.
        t0 = time.perf_counter()
        cache_text = chosen_hit.best_prop or chosen_hit.text
        cache_emb = np.asarray(embed_text(cache_text), dtype=np.float32)
        timings["cache_add_embed"] = (time.perf_counter() - t0) * 1000
        cache.add(
            dialogue_id,
            hit=chosen_hit,
            embedding=cache_emb,
            turn=turn_number,
            q2_score=final_q2,
        )

    cache_out = CacheOut(
        dialogue_id=dialogue_id,
        hit=cache_lookup.hit,
        similarity=float(cache_lookup.similarity),
        tau_cache=req.settings.tau_cache,
        cached_title=(cache_lookup.entry.hit.title if cache_lookup.hit and cache_lookup.entry else None),
        cached_turn=(cache_lookup.entry.turn if cache_lookup.hit and cache_lookup.entry else None),
        size_after=cache.size(dialogue_id),
        max_size=MAX_ENTRIES_PER_DIALOGUE,
        flare_cache_hit=flare_cache_hit,
    )

    return RunResponse(
        rewrite=RewriteOut(rewrite=rw.rewrite, paraphrases=rw.paraphrases),
        dense=[
            DenseHitOut(
                rank=i + 1,
                title=h.title,
                text=h.text,
                best_prop=h.best_prop,
                dense_score=float(h.score),
            )
            for i, h in enumerate(hits)
        ],
        reranked=[
            RerankedHitOut(
                rank=i + 1,
                title=h.title,
                text=h.text,
                best_prop=h.best_prop,
                dense_score=float(h.dense_score),
                rerank_score=float(h.rerank_score),
            )
            for i, h in enumerate(reranked)
        ],
        selection=selection_out,
        kedit=kedit_out,
        flare=flare_out,
        faithfulness=faithfulness_out,
        cache=cache_out,
        dialogue_id=dialogue_id,
        answer=final_answer,
        used_titles=final_used_titles,
        prompt=final_prompt,
        timings_ms={k: round(v, 1) for k, v in timings.items()},
    )
