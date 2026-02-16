"""Query Strategist — generates search variations from a user query.

Takes a query + context, produces 1-3 search parameter variations
using a small model. Decides which Bodega API to target.
"""

from __future__ import annotations


class QueryStrategist:
    """Phase 1 stub. Full implementation in Session 3 (Web Agent)."""

    async def strategize(self, query: str, context: dict | None = None) -> list[dict[str, str]]:
        """Generate search strategies for the given query."""
        return [{"query": query, "api": "search"}]
