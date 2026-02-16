"""OSA Evaluator — reviews assembled results, quality gate.

Phase 1: Simple pass-through (concatenate outputs).
Phase 2+: LLM-powered quality scoring and re-execution triggers.
"""

from __future__ import annotations

import structlog

from octane.models.schemas import AgentResponse

logger = structlog.get_logger().bind(component="osa.evaluator")


class Evaluator:
    """Reviews assembled agent outputs and produces final response.

    Phase 1: Concatenate successful outputs.
    Phase 2+: Use big model to synthesize, score quality, trigger re-execution.
    """

    async def evaluate(self, original_query: str, results: list[AgentResponse]) -> str:
        """Assemble agent results into a final output.

        Args:
            original_query: The user's original query
            results: Responses from all dispatched agents

        Returns:
            Final output string for the user
        """
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        if failed:
            for f in failed:
                logger.warning(
                    "agent_failed",
                    agent=f.agent,
                    error=f.error,
                )

        if not successful:
            return "⚠ All agents failed. Please try again."

        # Phase 1: Just return the first successful output
        # Phase 2+: Synthesize across multiple agent outputs using LLM
        output_parts = [r.output for r in successful if r.output]

        if len(output_parts) == 1:
            return output_parts[0]

        # Multiple outputs — join with separators
        return "\n\n".join(output_parts)
