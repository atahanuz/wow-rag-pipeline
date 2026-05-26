"""Stage 4: Active Re-retrieval (FLARE, modern variant).

Original FLARE-direct (Jiang et al., 2023) inspects per-token probabilities of
the generated draft. Tokens below theta=0.6 are "low-confidence"; the remaining
high-confidence tokens form an implicit retrieval query; new passages are
fetched and the sentence is regenerated.

Our LLM endpoint does not reliably expose per-token logprobs, so we replace
the token-probability signal with an explicit LLM-based grounding check on the
draft answer:
    - Given the draft and the source knowledge, identify which claims in the
      draft are NOT supported by the knowledge.
    - If any are unsupported, those claims become the retrieval query and the
      pipeline does ONE refinement pass (re-retrieve, optionally rerank,
      regenerate with the original + new knowledge merged).

This module implements only the grounding-check + query-extraction step.
Re-retrieval and regeneration orchestration live in server.py — keeps the
single-orchestration-point convention.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from api_clients import achat, chat


@dataclass
class GroundingCheck:
    grounded: bool                  # True if every claim in the draft is supported by the knowledge
    unsupported_claims: list[str]   # claims the LLM flagged as not grounded
    retrieval_query: str            # built from unsupported_claims; "" if not triggered
    prompt: str                     # full LLM prompt (UI transparency)
    fallback: bool                  # True if JSON parsing failed -> treat as grounded


_SYSTEM = (
    "You are a strict fact-checker for a knowledge-grounded dialogue system. "
    "Given a draft answer and the source knowledge it was supposed to be based on, "
    "identify any specific claims in the draft that are NOT supported by the "
    "source knowledge. Only flag a claim as unsupported if it contains specific "
    "facts (names, dates, numbers, attributions, relationships) that are absent "
    "from the source. Do not flag general framing, opinions, or rephrasings. "
    "Output strict JSON."
)


def _build_prompt(user_turn: str, draft_answer: str, knowledge_text: str) -> str:
    return (
        f"User's question: {user_turn}\n\n"
        f"Source knowledge:\n{knowledge_text}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        'Return JSON: {\n'
        '  "grounded": true | false,\n'
        '  "unsupported_claims": ["<verbatim or near-verbatim claim from the draft>", "..."]\n'
        "}\n"
        'If every specific claim in the draft is supported by the source knowledge, return '
        '{"grounded": true, "unsupported_claims": []}. '
        "Otherwise list each unsupported claim as a short phrase."
    )


def _parse(raw: str) -> tuple[bool, list[str], bool]:
    """Returns (grounded, unsupported_claims, fallback_flag)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return True, [], True
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return True, [], True

    grounded = bool(obj.get("grounded", True))
    raw_claims = obj.get("unsupported_claims", [])
    if not isinstance(raw_claims, list):
        return grounded, [], True
    claims = [str(c).strip() for c in raw_claims if str(c).strip()]
    # consistency: if claims is non-empty, grounded must be False
    if claims:
        grounded = False
    return grounded, claims, False


def _build_query(unsupported_claims: list[str], user_turn: str) -> str:
    """Form a retrieval query from the unsupported claims, optionally seeded with the user turn."""
    body = " ".join(unsupported_claims).strip()
    if not body:
        return ""
    # include the user turn for topical context — short queries do better on dense retrieval
    return f"{user_turn} {body}".strip()


def check(
    user_turn: str,
    draft_answer: str,
    knowledge_text: str,
    temperature: float = 0.0,
) -> GroundingCheck:
    if not draft_answer.strip() or not knowledge_text.strip():
        return GroundingCheck(grounded=True, unsupported_claims=[], retrieval_query="", prompt="", fallback=False)
    prompt = _build_prompt(user_turn, draft_answer, knowledge_text)
    raw = chat(prompt, system=_SYSTEM, temperature=temperature)
    grounded, unsupported, fallback = _parse(raw)
    query = _build_query(unsupported, user_turn) if (not grounded and unsupported) else ""
    return GroundingCheck(
        grounded=grounded,
        unsupported_claims=unsupported,
        retrieval_query=query,
        prompt=prompt,
        fallback=fallback,
    )


async def acheck(
    user_turn: str,
    draft_answer: str,
    knowledge_text: str,
    temperature: float = 0.0,
) -> GroundingCheck:
    if not draft_answer.strip() or not knowledge_text.strip():
        return GroundingCheck(grounded=True, unsupported_claims=[], retrieval_query="", prompt="", fallback=False)
    prompt = _build_prompt(user_turn, draft_answer, knowledge_text)
    raw = await achat(prompt, system=_SYSTEM, temperature=temperature)
    grounded, unsupported, fallback = _parse(raw)
    query = _build_query(unsupported, user_turn) if (not grounded and unsupported) else ""
    return GroundingCheck(
        grounded=grounded,
        unsupported_claims=unsupported,
        retrieval_query=query,
        prompt=prompt,
        fallback=fallback,
    )
