"""Stage: grounded answer generation (placeholder for the eventual GenKS+KEDiT stack).

For now this is plain RAG: feed the top-K reranked passages + dialogue history
+ user turn into gpt-oss-120b and have it produce a conversational answer that
sticks to the passages. Later this will be replaced by:
    - GenKS (Stage 2) — picks one snippet identifier
    - KEDiT (Stage 3) — Q-Former + KA-Adapter
    - FLARE (Stage 4) — active re-retrieval mid-decoding
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from api_clients import achat, chat
from reranker import RerankedHit


@dataclass
class GenerationResult:
    answer: str
    used_passages: list[RerankedHit]
    prompt: str             # exposed for transparency / debugging in the UI


_SYSTEM = (
    "You are a knowledge-grounded conversational assistant. "
    "Answer the user's latest turn using ONLY the provided knowledge passages. "
    "Rules:\n"
    "  - If the passages do not contain the answer, say so plainly. Do not invent facts.\n"
    "  - Keep responses conversational (1-3 sentences typically) and natural to the dialogue.\n"
    "  - Prefer the highest-ranked passage when multiple cover the same fact.\n"
    "  - Do not mention passage numbers or that you were given passages — just answer."
)


def _format_history(history: Sequence[tuple[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{role.capitalize()}: {text}" for role, text in history)


def _format_passages(passages: Sequence[RerankedHit]) -> str:
    if not passages:
        return "(no passages provided)"
    blocks = []
    for i, h in enumerate(passages, 1):
        blocks.append(f"[{i}] {h.title}\n{h.text}")
    return "\n\n".join(blocks)


def _build_prompt(user_turn: str, history: Sequence[tuple[str, str]], passages: Sequence[RerankedHit]) -> str:
    return (
        f"Knowledge passages:\n{_format_passages(passages)}\n\n"
        f"Dialogue history:\n{_format_history(history)}\n\n"
        f"User: {user_turn}\n"
        f"Assistant:"
    )


def generate(
    user_turn: str,
    history: Sequence[tuple[str, str]],
    passages: Sequence[RerankedHit],
    temperature: float = 0.2,
) -> GenerationResult:
    prompt = _build_prompt(user_turn, history, passages)
    answer = chat(prompt, system=_SYSTEM, temperature=temperature)
    return GenerationResult(answer=answer.strip(), used_passages=list(passages), prompt=prompt)


async def agenerate(
    user_turn: str,
    history: Sequence[tuple[str, str]],
    passages: Sequence[RerankedHit],
    temperature: float = 0.2,
) -> GenerationResult:
    prompt = _build_prompt(user_turn, history, passages)
    answer = await achat(prompt, system=_SYSTEM, temperature=temperature)
    return GenerationResult(answer=answer.strip(), used_passages=list(passages), prompt=prompt)
