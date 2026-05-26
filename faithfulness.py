"""Stage 5: Faithfulness Verification (Q² + BEGIN, modern variant).

Originals:
  - Q² (Honovich et al., 2021): T5-base QG + Albert-Xlarge QA + RoBERTa NLI
    pipeline that scores response/knowledge consistency via question-answer
    cross-checking.
  - BEGIN (Dziri et al., 2022): RoBERTa-base classifier fine-tuned on
    BEGIN-Adversarial, outputs {Fully Attributable, Not Attributable, Generic}.

We replace the four fine-tuned models with two LLM calls to gpt-oss-120b that
emulate the same protocols. Both checks run in parallel via asyncio.gather.

Gate verdict:
    failed if (q2_score < q2_threshold) OR (begin_label == "Not Attributable")
    -> server.py regenerates once using the GenKS runner-up passage.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from api_clients import achat


# ---------------------------------------------------------------------------
# Q²
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    question: str
    response_answer: str
    knowledge_answer: str | None      # None if knowledge does not address the question
    match: bool                       # response_answer agrees with knowledge_answer


@dataclass
class Q2Result:
    score: float                      # fraction of qa_pairs with match=True (1.0 if no pairs)
    qa_pairs: list[QAPair]
    prompt: str
    fallback: bool


_Q2_SYSTEM = (
    "You implement the Q² faithfulness check for a knowledge-grounded dialogue "
    "system. Given a response and a source knowledge passage, you generate "
    "questions whose answer is each substantive factual claim in the response, "
    "then verify whether each question can be answered with the same answer "
    "using only the knowledge. Output strict JSON."
)


def _q2_prompt(response: str, knowledge: str) -> str:
    return (
        f"Source knowledge:\n{knowledge}\n\n"
        f"Response:\n{response}\n\n"
        "For each substantive factual claim in the response (2 to 5 claims), "
        "generate one question whose answer IS that claim. Then try to answer "
        "the same question using ONLY the knowledge. Mark match=true only when "
        "the knowledge-based answer agrees with the response's answer in factual "
        "content (paraphrase is fine; missing facts is not).\n\n"
        'Return JSON: {\n'
        '  "qa_pairs": [\n'
        '    {"question": "...", '
        '"response_answer": "...", '
        '"knowledge_answer": "..." or null, '
        '"match": true|false},\n'
        '    ...\n'
        '  ]\n'
        "}\n"
        'If the response makes no substantive factual claims (e.g., generic '
        'acknowledgment, small talk), return {"qa_pairs": []}.'
    )


def _extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _parse_q2(raw: str) -> tuple[Q2Result, bool]:
    obj = _extract_json(raw)
    if obj is None or not isinstance(obj.get("qa_pairs"), list):
        return Q2Result(score=1.0, qa_pairs=[], prompt="", fallback=True), True
    pairs: list[QAPair] = []
    for x in obj["qa_pairs"]:
        if not isinstance(x, dict):
            continue
        q = str(x.get("question", "")).strip()
        ra = str(x.get("response_answer", "")).strip()
        ka_raw = x.get("knowledge_answer")
        ka = None if ka_raw is None else str(ka_raw).strip()
        match = bool(x.get("match", False))
        if q and ra:
            pairs.append(QAPair(question=q, response_answer=ra, knowledge_answer=ka, match=match))
    if not pairs:
        return Q2Result(score=1.0, qa_pairs=[], prompt="", fallback=False), False
    matches = sum(1 for p in pairs if p.match)
    return Q2Result(score=matches / len(pairs), qa_pairs=pairs, prompt="", fallback=False), False


async def aq2(response: str, knowledge: str) -> Q2Result:
    if not response.strip() or not knowledge.strip():
        return Q2Result(score=1.0, qa_pairs=[], prompt="", fallback=False)
    prompt = _q2_prompt(response, knowledge)
    raw = await achat(prompt, system=_Q2_SYSTEM, temperature=0.0)
    result, _ = _parse_q2(raw)
    result.prompt = prompt
    return result


# ---------------------------------------------------------------------------
# BEGIN
# ---------------------------------------------------------------------------

@dataclass
class BeginResult:
    label: str                # "Fully Attributable", "Not Attributable", or "Generic"
    rationale: str
    prompt: str
    fallback: bool


_BEGIN_LABELS = {"Fully Attributable", "Not Attributable", "Generic"}

_BEGIN_SYSTEM = (
    "You classify dialogue-system responses for knowledge attribution using the "
    "BEGIN taxonomy. Output strict JSON with exactly one of the allowed labels."
)


def _begin_prompt(response: str, knowledge: str) -> str:
    return (
        f"Source knowledge:\n{knowledge}\n\n"
        f"Response:\n{response}\n\n"
        "Classify the response as exactly ONE of:\n"
        "  \"Fully Attributable\": every specific factual claim in the response is "
        "supported by the knowledge.\n"
        "  \"Not Attributable\": the response contains specific facts (names, dates, "
        "numbers, attributions) that are NOT supported by the knowledge.\n"
        "  \"Generic\": the response is a generic conversational turn without specific "
        "factual claims (acknowledgment, small talk, opinions).\n\n"
        'Return JSON: {"label": "<one of: Fully Attributable | Not Attributable | Generic>", '
        '"rationale": "<one sentence justification>"}'
    )


def _parse_begin(raw: str) -> tuple[BeginResult, bool]:
    obj = _extract_json(raw)
    if obj is None:
        return BeginResult(label="Generic", rationale="(parse failed)", prompt="", fallback=True), True
    label = str(obj.get("label", "")).strip()
    rationale = str(obj.get("rationale", "")).strip()
    if label not in _BEGIN_LABELS:
        return BeginResult(label="Generic", rationale=f"(unknown label: {label!r}) {rationale}", prompt="", fallback=True), True
    return BeginResult(label=label, rationale=rationale, prompt="", fallback=False), False


async def abegin(response: str, knowledge: str) -> BeginResult:
    if not response.strip() or not knowledge.strip():
        return BeginResult(label="Generic", rationale="empty input", prompt="", fallback=False)
    prompt = _begin_prompt(response, knowledge)
    raw = await achat(prompt, system=_BEGIN_SYSTEM, temperature=0.0)
    result, _ = _parse_begin(raw)
    result.prompt = prompt
    return result


# ---------------------------------------------------------------------------
# Combined gate
# ---------------------------------------------------------------------------

@dataclass
class FaithfulnessResult:
    q2: Q2Result
    begin: BeginResult
    failed: bool                  # gate verdict (regeneration trigger)
    q2_threshold: float


async def aevaluate(
    response: str,
    knowledge: str,
    q2_threshold: float = 0.5,
) -> FaithfulnessResult:
    q2_result, begin_result = await asyncio.gather(
        aq2(response, knowledge),
        abegin(response, knowledge),
    )
    failed = (q2_result.score < q2_threshold) or (begin_result.label == "Not Attributable")
    return FaithfulnessResult(
        q2=q2_result,
        begin=begin_result,
        failed=failed,
        q2_threshold=q2_threshold,
    )


def evaluate(
    response: str,
    knowledge: str,
    q2_threshold: float = 0.5,
) -> FaithfulnessResult:
    return asyncio.run(aevaluate(response, knowledge, q2_threshold))
