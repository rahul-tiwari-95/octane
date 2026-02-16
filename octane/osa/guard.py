"""OSA Guard — input/output safety checks.

Hybrid: regex for injection patterns, basic checks for safety.
Phase 2+: Small model for semantic safety checks.
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger().bind(component="osa.guard")


class Guard:
    """Input and output safety validation.

    Phase 1: Basic regex patterns and length checks.
    Phase 2+: Small model for semantic analysis.
    """

    # Basic injection patterns to watch for
    INJECTION_PATTERNS: list[str] = [
        r"ignore\s+(?:all\s+)?(?:previous|above)\s+instructions",
        r"you\s+are\s+now\s+(?:a|an)\s+(?:different|new)",
        r"system\s*:\s*",
        r"<\s*script\s*>",
    ]

    MAX_INPUT_LENGTH: int = 10000

    async def check_input(self, query: str) -> dict[str, bool | str]:
        """Validate user input for safety.

        Returns:
            {"safe": True/False, "reason": "..." if blocked}
        """
        # Length check
        if len(query) > self.MAX_INPUT_LENGTH:
            return {"safe": False, "reason": f"Query too long ({len(query)} chars, max {self.MAX_INPUT_LENGTH})"}

        # Empty check
        if not query.strip():
            return {"safe": False, "reason": "Empty query"}

        # Injection pattern check
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("injection_detected", pattern=pattern, query_preview=query[:100])
                return {"safe": False, "reason": "Potential prompt injection detected"}

        return {"safe": True}

    async def check_output(self, output: str) -> dict[str, bool | str]:
        """Validate agent output for safety (Phase 1: pass-through)."""
        if not output:
            return {"safe": True}

        return {"safe": True}
