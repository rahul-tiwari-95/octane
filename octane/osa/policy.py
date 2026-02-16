"""OSA Policy Engine — deterministic rules.

No LLM. Pure Python logic for:
- Max retries
- HITL (human-in-the-loop) triggers
- Allowed/forbidden actions
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="osa.policy")


class PolicyEngine:
    """Deterministic rules engine for OSA.

    All rules are pure Python — no LLM inference.
    """

    MAX_RETRIES: int = 3
    MAX_QUERY_LENGTH: int = 10000
    DESTRUCTIVE_KEYWORDS: list[str] = ["delete", "remove", "drop", "destroy", "format"]

    def check_retries(self, current_retries: int) -> bool:
        """Check if retries are within policy limits."""
        return current_retries < self.MAX_RETRIES

    def requires_confirmation(self, query: str) -> bool:
        """Check if the query requires human confirmation."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.DESTRUCTIVE_KEYWORDS)

    def check_query_length(self, query: str) -> bool:
        """Check if query is within length limits."""
        return len(query) <= self.MAX_QUERY_LENGTH
