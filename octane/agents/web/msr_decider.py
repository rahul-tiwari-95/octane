"""Multi-Shot Refinement (MSR) Decider.

After a Round-1 scout search, MSRDecider reads the initial findings and
decides whether the user's query is ambiguous enough to benefit from
targeted clarifying questions before running the deep-dive.

Design principles:
- Always run a full Round-1 scout first so Octane has actual evidence
  before deciding whether clarification is needed.
- Use ModelTier.FAST (90M) — this is a classification/generation task,
  not deep reasoning.  Speed matters here so the user doesn't wait long.
- Generate multiple-choice questions so the user can answer with a single
  key press (A/B/C/D) — low friction, high signal.
- Return at most 3 questions to keep the interaction tight.

MSR fires when the query is:
  - Ambiguous (multiple valid interpretations with different research directions)
  - Multi-faceted (user could want military, diplomatic, economic, or political angle)

MSR skips when:
  - Query is specific and unambiguous ("what is NVDA stock price")
  - Round-1 findings already clearly answer the query
  - LLM decides clarification adds no value
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import structlog

from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="web.msr_decider")

_SENTINEL = object()

_MSR_SYSTEM = """\
You are a research strategist. A user has submitted a query and Octane has \
completed a first-pass scout search. Your task is to decide whether the query \
is ambiguous or multi-faceted enough that targeted clarification questions would \
significantly improve the research direction.

Decision criteria:
- SHOULD ASK: Query has 2+ meaningfully different research angles. \
  The options must be specific to the query topic — derive them from the \
  query content and scout findings, not from a generic template.
- SHOULD SKIP: Query is specific, factual, or clear enough that clarification \
  adds no value (e.g. stock prices, single-event queries, named entities).

If you decide to ask, generate 1-3 multiple-choice questions. Each question \
must have 3-4 answer options that represent genuinely different research angles \
RELEVANT TO THE SPECIFIC QUERY. Do not reuse generic geopolitical categories \
(war/diplomacy/economy) unless the query is explicitly about international events. \
Derive all options from the actual topic — spiritual, scientific, financial, \
cultural, or otherwise.

Return ONLY valid JSON in this exact format:
{
  "should_ask": true,
  "questions": [
    {
      "text": "<question text relevant to the query>",
      "options": ["<option A>", "<option B>", "<option C>", "<option D>"]
    }
  ]
}

If should_ask is false, return:
{"should_ask": false, "questions": []}

No prose. No markdown. Return ONLY the JSON object."""


@dataclass
class MSRQuestion:
    """A single multiple-choice clarification question."""
    text: str
    options: list[str] = field(default_factory=list)


@dataclass
class MSRResult:
    """Result of the MSR decision process."""
    should_ask: bool
    questions: list[MSRQuestion] = field(default_factory=list)


class MSRDecider:
    """Decides whether to ask clarifying MCQ questions after Round-1 scout.

    Args:
        bodega: BodegaRouter instance. Pass None to always skip (no-op mode).
    """

    def __init__(self, bodega=_SENTINEL) -> None:
        if bodega is _SENTINEL:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega

    async def decide(
        self,
        query: str,
        findings: list[str],
        max_questions: int = 3,
    ) -> MSRResult:
        """Analyse Round-1 findings and decide whether to ask clarifying questions.

        Args:
            query:         The user's original query.
            findings:      Text excerpts from Round-1 extracted pages (used as
                           evidence that the query is or isn't ambiguous).
            max_questions: Cap on questions generated (default 3).

        Returns:
            MSRResult — should_ask=False if no clarification needed, else
            should_ask=True with 1-3 MSRQuestion instances.
        """
        if self._bodega is None or not findings:
            logger.debug("msr_skipped", reason="no_bodega_or_no_findings")
            return MSRResult(should_ask=False)

        try:
            return await self._llm_decide(query, findings, max_questions)
        except Exception as exc:
            logger.warning("msr_decision_failed", error=str(exc))
            return MSRResult(should_ask=False)

    async def _llm_decide(
        self,
        query: str,
        findings: list[str],
        max_questions: int,
    ) -> MSRResult:
        """Use FAST-tier LLM to decide ambiguity and generate questions."""
        # Condense findings to avoid context overflow
        condensed = "\n".join(
            f"- {f[:200]}" for f in findings[:6] if f and f.strip()
        )

        prompt = (
            f'User query: "{query}"\n\n'
            f"Round-1 scout findings (excerpts):\n{condensed}\n\n"
            f"Decide: is this query ambiguous or multi-faceted enough to benefit "
            f"from clarification questions? If yes, generate up to {max_questions} "
            f"multiple-choice questions."
        )

        raw = await asyncio.wait_for(
            self._bodega.chat_simple(
                prompt=prompt,
                system=_MSR_SYSTEM,
                tier=ModelTier.FAST,
                temperature=0.3,
                max_tokens=1024,
            ),
            timeout=30.0,
        )

        # Strip <think> block if present
        if "</think>" in raw:
            think_part, _, clean = raw.partition("</think>")
            logger.debug(
                "msr_reasoning",
                trace=think_part.replace("<think>", "").strip()[:300],
            )
        else:
            clean = raw

        # Extract JSON object
        json_match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not json_match:
            logger.warning("msr_no_json", response_preview=clean[:120])
            return MSRResult(should_ask=False)

        data = json.loads(json_match.group(0))

        should_ask = bool(data.get("should_ask", False))
        questions_raw = data.get("questions", [])

        if not should_ask or not questions_raw:
            logger.info("msr_decision_skip", query=query[:60])
            return MSRResult(should_ask=False)

        questions: list[MSRQuestion] = []
        for q in questions_raw[:max_questions]:
            if isinstance(q, dict) and "text" in q:
                opts = [str(o) for o in q.get("options", [])[:4]]
                if opts:  # only keep questions that have options
                    questions.append(MSRQuestion(text=str(q["text"]), options=opts))

        if not questions:
            return MSRResult(should_ask=False)

        logger.info(
            "msr_decision_ask",
            query=query[:60],
            n_questions=len(questions),
            questions=[q.text[:60] for q in questions],
        )
        return MSRResult(should_ask=True, questions=questions)
