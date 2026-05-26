"""Extension 4: Conversation-aware query rewriter.

Given a dialogue history and the current user turn, produce:
  - a self-contained rewrite where pronouns/coreferences are resolved
  - K-1 semantic paraphrases of that rewrite

Default K=3 → one rewrite + two paraphrases (matches proposal example).

Also exposes Reciprocal Rank Fusion (RRF) so callers can fuse retrieval lists
issued for each query variant.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from api_clients import achat, chat


@dataclass
class RewriteResult:
    rewrite: str
    paraphrases: list[str]

    @property
    def all_queries(self) -> list[str]:
        return [self.rewrite, *self.paraphrases]


_SYSTEM = (
    "You rewrite conversational user turns into self-contained search queries. "
    "Resolve every pronoun and implicit reference using the dialogue history. "
    "Output strictly valid JSON."
)


def _format_history(history: Sequence[tuple[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{role}: {text}" for role, text in history)


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object found in model output: {raw!r}")
    return json.loads(m.group(0))


def _build_prompt(user_turn: str, history: Sequence[tuple[str, str]], num_paraphrases: int) -> str:
    return (
        f"Dialogue history:\n{_format_history(history)}\n\n"
        f"Current user turn: {user_turn}\n\n"
        f"Return JSON with two fields:\n"
        f'  "rewrite": a single self-contained version of the user turn that resolves all '
        f"pronouns and implicit references using the history. Keep it terse and search-friendly.\n"
        f'  "paraphrases": a list of exactly {num_paraphrases} semantic paraphrases of the rewrite '
        f"that use different wording but ask for the same information.\n"
        f"Output JSON only."
    )


def _parse_rewrite(raw: str, num_paraphrases: int) -> RewriteResult:
    obj = _extract_json(raw)
    rw = str(obj["rewrite"]).strip()
    paras = [str(p).strip() for p in obj.get("paraphrases", []) if str(p).strip()][:num_paraphrases]
    return RewriteResult(rewrite=rw, paraphrases=paras)


def rewrite(
    user_turn: str,
    history: Sequence[tuple[str, str]] = (),
    num_paraphrases: int = 2,
) -> RewriteResult:
    raw = chat(_build_prompt(user_turn, history, num_paraphrases), system=_SYSTEM, temperature=0.0)
    return _parse_rewrite(raw, num_paraphrases)


async def arewrite(
    user_turn: str,
    history: Sequence[tuple[str, str]] = (),
    num_paraphrases: int = 2,
) -> RewriteResult:
    raw = await achat(_build_prompt(user_turn, history, num_paraphrases), system=_SYSTEM, temperature=0.0)
    return _parse_rewrite(raw, num_paraphrases)


def rrf_fuse(ranked_lists: Sequence[Sequence], k: int = 60) -> list:
    """Reciprocal Rank Fusion. Each input is an ordered list of hashable ids.

    Returns ids sorted by aggregate RRF score (descending).
    """
    scores: dict = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


if __name__ == "__main__":
    history = [
        ("user", "I love the Eiffel Tower."),
        ("bot", "It is a famous landmark in Paris."),
    ]
    result = rewrite("When was it built?", history=history, num_paraphrases=2)
    print("rewrite     :", result.rewrite)
    for i, p in enumerate(result.paraphrases, 1):
        print(f"paraphrase {i}:", p)
