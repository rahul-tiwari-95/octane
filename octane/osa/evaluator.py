"""OSA Evaluator — reviews assembled results, synthesizes final output.

Session 2: LLM-powered synthesis using Bodega Inference.
Falls back to simple concatenation if Bodega is unavailable.

Phase 3+: Quality scoring, confidence gates, re-execution triggers.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

import structlog

from octane.models.schemas import AgentResponse
from octane.tools.topology import ModelTier
from octane.utils.clock import today_human

logger = structlog.get_logger().bind(component="osa.evaluator")

# Sentinel: distinguish "use default BodegaRouter" from "explicitly no LLM"
_UNSET = object()

_EVALUATOR_SYSTEM_BASE = """\
You are the Octane Evaluator. Given the results from one or more specialized \
agents, synthesize a single clear, direct response to the user's original query.

Rules:
- Ground every claim in the data provided. Do not invent information.
- If an agent failed or returned no useful data, acknowledge it briefly \
  and answer with what you have.
- Do not mention "agents", "tools", or internal system names in your response.
- Respond directly to the user as if you are one coherent assistant.
- If the data contains a date that is not today, note that it may be outdated."""


def _build_system_prompt(user_profile: dict | None) -> str:
    """Build a personalized system prompt from the user's preference profile."""
    # Always ground the LLM in the current wall-clock date so it can flag
    # stale data (e.g. a stock price from 4 months ago) accurately.
    lines = [_EVALUATOR_SYSTEM_BASE, f"\nToday's date: {today_human()}."]

    if user_profile:
        verbosity = user_profile.get("verbosity", "concise")
        expertise = user_profile.get("expertise", "advanced")
        style = user_profile.get("response_style", "prose")
        domains = user_profile.get("domains", "")

        style_instructions: list[str] = []

        if verbosity == "concise":
            style_instructions.append("Be brief and direct. No padding, no repetition.")
        elif verbosity == "detailed":
            style_instructions.append("Be thorough. Include supporting context and detail.")

        if expertise == "beginner":
            style_instructions.append("Use simple language. Avoid jargon. Explain terms.")
        elif expertise == "advanced":
            style_instructions.append("Assume technical fluency. Skip basic explanations.")

        if style == "bullets":
            style_instructions.append("Use bullet points for lists and multi-part answers.")
        elif style == "code-first":
            style_instructions.append("Lead with code examples where relevant.")

        if style_instructions:
            lines.append("\nUser style preferences:\n- " + "\n- ".join(style_instructions))

    return "\n".join(lines)


class Evaluator:
    """Reviews assembled agent outputs and synthesizes the final response.

    Session 2: Uses LLM to synthesize across agent results.
    Falls back to passthrough concatenation if Bodega is unavailable.
    """

    def __init__(self, bodega=_UNSET) -> None:
        # Pass bodega=None explicitly to disable LLM (concatenation fallback).
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega

    async def evaluate(
        self,
        original_query: str,
        results: list[AgentResponse],
        prior_context: str | None = None,
        user_profile: dict | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Assemble agent results into a final output.

        Args:
            original_query: The user's original query
            results: Responses from all dispatched agents
            prior_context: Optional prior memory context from MemoryAgent
            user_profile: Optional user preferences for tone/style
            conversation_history: Optional list of prior turns
                [{"role": "user"|"assistant", "content": "..."}]
                Injected directly into the LLM prompt for multi-turn continuity.

        Returns:
            Final synthesized output string for the user
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        for f in failed:
            logger.warning("agent_failed", agent=f.agent, error=f.error)

        if not successful:
            return "⚠ All agents failed to produce a response. Please try again."

        output_parts = [r.output for r in successful if r.output]

        if not output_parts:
            return "⚠ Agents responded but produced no output."

        # If Bodega is available, synthesize with LLM
        if self._bodega is not None:
            try:
                return await self._synthesize_with_llm(
                    original_query, output_parts, failed, prior_context, user_profile,
                    conversation_history=conversation_history,
                )
            except Exception as exc:
                logger.warning("llm_synthesis_failed", error=str(exc), fallback="concatenation")

        # Fallback: plain concatenation
        if len(output_parts) == 1:
            return output_parts[0]
        return "\n\n---\n\n".join(output_parts)

    async def _synthesize_with_llm(
        self,
        query: str,
        output_parts: list[str],
        failed: list[AgentResponse],
        prior_context: str | None = None,
        user_profile: dict | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Use Bodega to synthesize agent results into a cohesive response."""
        system = _build_system_prompt(user_profile)
        context_lines = []

        # Inject rolling conversation history for multi-turn continuity
        if conversation_history:
            history_text = _format_conversation_history(conversation_history)
            if history_text:
                context_lines.append(f"[Conversation history]\n{history_text}")

        if prior_context:
            context_lines.append(f"[Prior context from memory]\n{prior_context.strip()}")

        for i, part in enumerate(output_parts, 1):
            context_lines.append(f"[Result {i}]\n{part.strip()}")

        if failed:
            failed_names = ", ".join(r.agent for r in failed)
            context_lines.append(f"[Note: the following agents failed: {failed_names}]")

        context = "\n\n".join(context_lines)

        prompt = (
            f'User query: "{query}"\n\n'
            f"Agent results:\n{context}\n\n"
            f"Synthesize a direct response to the user's query."
        )

        response = await asyncio.wait_for(
            self._bodega.chat_simple(
                prompt=prompt,
                system=system,
                tier=ModelTier.REASON,
                temperature=0.3,
                max_tokens=512,  # research synthesis needs key facts, not long essays
            ),
            timeout=30.0,  # evaluator synthesis: 30 s cap
        )
        return response.strip()

    def _build_prompt(
        self,
        query: str,
        output_parts: list[str],
        failed: list[AgentResponse],
        prior_context: str | None = None,
    ) -> tuple[str, str]:
        """Build (system, user_prompt) for both streaming and non-streaming paths."""
        raise NotImplementedError  # unused — kept for symmetry

    async def evaluate_stream(
        self,
        original_query: str,
        results: list[AgentResponse],
        prior_context: str | None = None,
        user_profile: dict | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Like evaluate(), but yields text chunks as they stream from Bodega.

        Falls back to yielding the full evaluate() result in one chunk if
        streaming is unavailable (Bodega offline or no stream support).

        Args:
            conversation_history: Optional rolling buffer of prior turns for
                direct multi-turn context injection.

        Usage in CLI:
            async for chunk in evaluator.evaluate_stream(query, results):
                print(chunk, end="", flush=True)
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        for f in failed:
            logger.warning("agent_failed", agent=f.agent, error=f.error)

        if not successful:
            yield "⚠ All agents failed to produce a response. Please try again."
            return

        output_parts = [r.output for r in successful if r.output]
        if not output_parts:
            yield "⚠ Agents responded but produced no output."
            return

        if self._bodega is None:
            # No LLM — yield concatenated output in one shot
            yield output_parts[0] if len(output_parts) == 1 else "\n\n---\n\n".join(output_parts)
            return

        # Build prompt — inject conversation history + prior memory context
        system = _build_system_prompt(user_profile)
        context_lines = []

        if conversation_history:
            history_text = _format_conversation_history(conversation_history)
            if history_text:
                context_lines.append(f"[Conversation history]\n{history_text}")

        if prior_context:
            context_lines.append(f"[Prior context from memory]\n{prior_context.strip()}")
        for i, part in enumerate(output_parts, 1):
            context_lines.append(f"[Result {i}]\n{part.strip()}")
        if failed:
            context_lines.append(f"[Note: the following agents failed: {', '.join(r.agent for r in failed)}]")

        context = "\n\n".join(context_lines)
        prompt = (
            f'User query: "{original_query}"\n\n'
            f"Agent results:\n{context}\n\n"
            f"Synthesize a direct response to the user's query."
        )

        try:
            # Stream tokens, stripping <think>...</think> blocks inline.
            # Strategy: buffer everything until we've seen </think> (or confirmed
            # no think block exists), then flush the clean remainder.
            raw_buffer = ""
            think_done = False

            async for chunk in self._bodega.chat_stream(
                prompt=prompt,
                system=system,
                tier=ModelTier.REASON,
                temperature=0.3,
                max_tokens=512,  # research synthesis: key facts only, not essays
            ):
                raw_buffer += chunk

                if not think_done:
                    # Still waiting to determine if there's a think block
                    if "</think>" in raw_buffer:
                        # Strip the entire think block and start streaming
                        clean = re.sub(r"<think>.*?</think>", "", raw_buffer, flags=re.DOTALL).lstrip()
                        think_done = True
                        if clean:
                            yield clean
                            raw_buffer = ""
                    elif "<think>" not in raw_buffer and len(raw_buffer) > 20:
                        # No think block started after 20 chars — safe to stream
                        think_done = True
                        yield raw_buffer
                        raw_buffer = ""
                    # else: still accumulating — keep buffering
                else:
                    # think block is done — stream chunks directly
                    yield chunk
                    raw_buffer = ""

            # Flush any remaining buffered content
            if raw_buffer:
                clean = re.sub(r"<think>.*?</think>", "", raw_buffer, flags=re.DOTALL).strip()
                if clean:
                    yield clean

        except Exception as exc:
            logger.warning("stream_failed", error=str(exc), fallback="evaluate()")
            # Graceful fallback to non-streaming
            try:
                result = await self.evaluate(
                    original_query, results,
                    prior_context=prior_context,
                    user_profile=user_profile,
                    conversation_history=conversation_history,
                )
            except Exception as eval_exc:
                logger.warning(
                    "evaluate_fallback_failed",
                    error=str(eval_exc),
                    fallback="concatenation",
                )
                # Last resort: plain concatenation so we always yield something
                successful_outputs = [r.output for r in results if r.success and r.output]
                result = "\n\n---\n\n".join(successful_outputs) if successful_outputs else (
                    "\n\n---\n\n".join(r.output for r in results if r.output)
                )
            yield result


# ── Module-level helpers ──────────────────────────────────────────────────────


def _format_conversation_history(
    history: list[dict[str, str]],
    max_turns: int = 6,
) -> str:
    """Format the last N turns of conversation history for LLM injection.

    Keeps the most recent `max_turns` entries. Each entry is formatted as
    'User: ...' or 'Assistant: ...' on its own line, capped at 300 chars each.
    """
    recent = history[-max_turns:]
    lines: list[str] = []
    for entry in recent:
        role = entry.get("role", "user")
        content = entry.get("content", "")[:300]
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)
