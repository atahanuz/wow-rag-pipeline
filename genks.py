"""Stage 2: Generative Knowledge Selection (GenKS, modern variant).

Original GenKS (Sun et al., 2023) fine-tunes BART-base to emit a snippet
identifier before the response. We replace the fine-tuned BART with a zero-shot
prompt to gpt-oss-120b that uses the same identifier scheme (k1, k2, ...).

Input : reranked top-K candidate passages
Output: exactly one chosen passage + a short reason

Downstream stages (generator, FLARE, faithfulness gate) see the single chosen
passage as the primary knowledge, with the runner-ups preserved for future use
(FLARE re-retrieval, regeneration on faithfulness failure).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from api_clients import achat, chat
from reranker import RerankedHit


@dataclass
class SelectionResult:
    chosen_idx: int           # 0-based index into the input list
    chosen_label: str         # "k1", "k2", ...
    chosen: RerankedHit
    reason: str
    prompt: str               # full prompt sent to the LLM (UI transparency)
    fallback: bool = False    # True when JSON parsing failed and we defaulted to k1


_SYSTEM = (
    "You are a knowledge selector for a dialogue system. "
    "Given a dialogue and a numbered list of candidate Wikipedia passages, "
    "pick the SINGLE most relevant passage that would let the assistant answer "
    "the user's latest turn correctly. Output strict JSON only."
)


def _format_history(history: Sequence[tuple[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{role.capitalize()}: {text}" for role, text in history)


def _format_candidates(hits: Sequence[RerankedHit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[k{i}] Title: {h.title}\n{h.text}")
    return "\n\n".join(blocks)


def _build_prompt(user_turn: str, history: Sequence[tuple[str, str]], hits: Sequence[RerankedHit]) -> str:
    labels = ", ".join(f"k{i + 1}" for i in range(len(hits)))
    return (
        f"Dialogue history:\n{_format_history(history)}\n\n"
        f"Current user turn: {user_turn}\n\n"
        f"Candidate passages:\n{_format_candidates(hits)}\n\n"
        f'Return JSON: {{"chosen": "<one of: {labels}>", "reason": "<one-sentence justification>"}}\n'
        "Choose exactly one. Do not invent identifiers. Do not return more than one identifier."
    )


def _parse(raw: str, n_hits: int) -> tuple[int, str, bool]:
    """Returns (chosen_idx_0based, reason, fallback_flag)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return 0, f"could not parse model output: {raw[:80]!r}", True
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return 0, f"JSON decode failed: {e}", True

    chosen = str(obj.get("chosen", "")).strip().lower()
    reason = str(obj.get("reason", "")).strip()

    m2 = re.match(r"k?(\d+)", chosen)
    if not m2:
        return 0, reason or f"unrecognized identifier: {chosen!r}", True
    one_based = int(m2.group(1))
    idx = one_based - 1
    if idx < 0 or idx >= n_hits:
        return 0, reason or f"identifier out of range: k{one_based}", True
    return idx, reason, False


def _trivial(hit: RerankedHit) -> SelectionResult:
    return SelectionResult(
        chosen_idx=0,
        chosen_label="k1",
        chosen=hit,
        reason="only one candidate",
        prompt="",
        fallback=False,
    )


def select(
    user_turn: str,
    history: Sequence[tuple[str, str]],
    hits: Sequence[RerankedHit],
    temperature: float = 0.0,
) -> SelectionResult:
    if not hits:
        raise ValueError("GenKS.select called with no candidates")
    if len(hits) == 1:
        return _trivial(hits[0])
    prompt = _build_prompt(user_turn, history, hits)
    raw = chat(prompt, system=_SYSTEM, temperature=temperature)
    idx, reason, fallback = _parse(raw, len(hits))
    return SelectionResult(
        chosen_idx=idx,
        chosen_label=f"k{idx + 1}",
        chosen=hits[idx],
        reason=reason,
        prompt=prompt,
        fallback=fallback,
    )


async def aselect(
    user_turn: str,
    history: Sequence[tuple[str, str]],
    hits: Sequence[RerankedHit],
    temperature: float = 0.0,
) -> SelectionResult:
    if not hits:
        raise ValueError("GenKS.aselect called with no candidates")
    if len(hits) == 1:
        return _trivial(hits[0])
    prompt = _build_prompt(user_turn, history, hits)
    raw = await achat(prompt, system=_SYSTEM, temperature=temperature)
    idx, reason, fallback = _parse(raw, len(hits))
    return SelectionResult(
        chosen_idx=idx,
        chosen_label=f"k{idx + 1}",
        chosen=hits[idx],
        reason=reason,
        prompt=prompt,
        fallback=fallback,
    )
