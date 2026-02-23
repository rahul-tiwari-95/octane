"""OSA Evaluator — reviews assembled results, synthesizes final output.

Session 2: LLM-powered synthesis using Bodega Inference.
Falls back to simple concatenation if Bodega is unavailable.

Phase 3+: Quality scoring, confidence gates, re-execution triggers.
"""

from __future__ import annotations

import structlog

from octane.models.schemas import AgentResponse

logger = structlog.get_logger().bind(component="osa.evaluator")

_EVALUATOR_SYSTEM = """\
You are the Octane Evaluator. Given the results from one or more specialized \
agents, synthesize a single clear, direct response to the user's original query.

Rules:
- Be concise. Do not pad or repeat yourself.
- Ground every claim in the data provided. Do not invent information.
- If an agent failed or returned no useful data, acknowledge it briefly \
  and answer with what you have.
- If multiple agents are working in parallel, their results may arrive at different times. \
  Be prepared to handle partial results and synthesize them as they come in.
- Do not mention "agents", "tools", or internal system names in your response.
- Respond directly to the user as if you are one coherent assistant."""


class Evaluator:
    """Reviews assembled agent outputs and synthesizes the final response.

    Session 2: Uses LLM to synthesize across agent results.
    Falls back to passthrough concatenation if Bodega is unavailable.
    """

    def __init__(self, bodega=None) -> None:
        self._bodega = bodega

    async def evaluate(
        self,
        original_query: str,
        results: list[AgentResponse],
        prior_context: str | None = None,
        user_profile: dict | None = None,
    ) -> str:
        """Assemble agent results into a final output.

        Args:
            original_query: The user's original query
            results: Responses from all dispatched agents
            prior_context: Optional prior memory context from MemoryAgent

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
                    original_query, output_parts, failed, prior_context, user_profile
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
    ) -> str:
        """Use Bodega to synthesize agent results into a cohesive response."""
        context_lines = []

        if user_profile:
            verbosity = user_profile.get("verbosity", "concise")
            expertise = user_profile.get("expertise", "advanced")
            style = user_profile.get("response_style", "prose")
            context_lines.append(
                f"[User preferences: verbosity={verbosity}, expertise={expertise}, style={style}]"
            )

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
            f"Synthesize a direct, concise response to the user's query."
        )

        response = await self._bodega.chat_simple(
            prompt=prompt,
            system=_EVALUATOR_SYSTEM,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.strip()
