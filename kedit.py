"""Stage 3: PEFT-style knowledge distillation (KEDiT, modern variant).

Original KEDiT (Zhang et al., 2025) compresses a passage into m=16 continuous
vectors via a BERT + Q-Former bottleneck trained on a 6M-chunk Wikipedia
dataset, then injects those vectors into a frozen Llama-3-8B through a
KA-Adapter (PEFT). Both stages require training and a local LLM.

We replace this with a zero-shot, query-conditioned distillation that preserves
KEDiT's *purpose* without the training stack:
    - Bottleneck (continuous m=16 vectors) -> a short, query-conditioned
      bullet list of facts extracted from the passage.
    - KA-Adapter (gated injection into the LLM) -> we simply place the
      distilled facts in the prompt; gpt-oss-120b is instruction-tuned and
      doesn't need adapter weights.

The generator now sees a tight, focused "knowledge brief" instead of the raw
paragraph. The original passage is preserved so the faithfulness gate
(Stage 5) can score against the actual source.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Sequence

from api_clients import achat, chat
from reranker import RerankedHit


@dataclass
class DistilledKnowledge:
    source: RerankedHit           # original passage chosen by GenKS
    summary: str                  # one-sentence summary of what this passage adds
    facts: list[str]              # 2-5 query-conditioned facts, faithful to source
    prompt: str                   # full LLM prompt (UI transparency)
    fallback: bool = False        # True if JSON parsing failed -> we fell back to source

    def as_brief(self) -> str:
        """Compact textual form for prompting the generator."""
        bullets = "\n".join(f"- {f}" for f in self.facts) if self.facts else self.source.text
        return f"{self.summary}\n{bullets}" if self.summary else bullets

    def to_passage(self) -> RerankedHit:
        """Drop-in RerankedHit whose `text` is the distilled brief.

        Generator API stays unchanged this way.
        """
        return replace(self.source, text=self.as_brief())


_SYSTEM = (
    "You distill Wikipedia passages into focused knowledge briefs for a "
    "dialogue assistant. Given a passage and the user's question, extract the "
    "SPECIFIC facts from the passage that are most relevant to answering the "
    "question. Do not invent facts. Only use information present in the "
    "passage. Output strict JSON."
)


def _format_history(history: Sequence[tuple[str, str]]) -> str:
    if not history:
        return "(no prior turns)"
    return "\n".join(f"{role.capitalize()}: {text}" for role, text in history)


def _build_prompt(user_turn: str, history: Sequence[tuple[str, str]], hit: RerankedHit) -> str:
    return (
        f"Title: {hit.title}\n"
        f"Passage:\n{hit.text}\n\n"
        f"Dialogue history:\n{_format_history(history)}\n\n"
        f"User's question: {user_turn}\n\n"
        'Return JSON with two fields:\n'
        '  "summary": "<one-sentence summary of what this passage contributes for answering the question>",\n'
        '  "facts": ["<fact 1 — short sentence, close to the passage wording>", "<fact 2>", "..."]\n'
        "Extract 2 to 5 facts. Each fact must be a self-contained sentence and "
        "must be supported by the passage. Do not paraphrase aggressively. "
        "Do not include any information that is not in the passage."
    )


def _parse(raw: str) -> tuple[str, list[str], bool]:
    """Returns (summary, facts, fallback_flag)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return "", [], True
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return "", [], True
    summary = str(obj.get("summary", "")).strip()
    raw_facts = obj.get("facts", [])
    if not isinstance(raw_facts, list):
        return summary, [], True
    facts = [str(f).strip() for f in raw_facts if str(f).strip()]
    return summary, facts[:5], False


def _from_source_fallback(hit: RerankedHit, prompt: str, reason: str) -> DistilledKnowledge:
    return DistilledKnowledge(
        source=hit,
        summary=reason,
        facts=[hit.text],
        prompt=prompt,
        fallback=True,
    )


def distill(
    user_turn: str,
    history: Sequence[tuple[str, str]],
    hit: RerankedHit,
    temperature: float = 0.0,
) -> DistilledKnowledge:
    prompt = _build_prompt(user_turn, history, hit)
    raw = chat(prompt, system=_SYSTEM, temperature=temperature)
    summary, facts, fallback = _parse(raw)
    if fallback or not facts:
        return _from_source_fallback(hit, prompt, summary or "(used raw passage)")
    return DistilledKnowledge(source=hit, summary=summary, facts=facts, prompt=prompt, fallback=False)


async def adistill(
    user_turn: str,
    history: Sequence[tuple[str, str]],
    hit: RerankedHit,
    temperature: float = 0.0,
) -> DistilledKnowledge:
    prompt = _build_prompt(user_turn, history, hit)
    raw = await achat(prompt, system=_SYSTEM, temperature=temperature)
    summary, facts, fallback = _parse(raw)
    if fallback or not facts:
        return _from_source_fallback(hit, prompt, summary or "(used raw passage)")
    return DistilledKnowledge(source=hit, summary=summary, facts=facts, prompt=prompt, fallback=False)
