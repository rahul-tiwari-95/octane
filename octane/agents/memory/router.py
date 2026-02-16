"""Memory Router — decides which tier to query."""

from __future__ import annotations


class MemoryRouter:
    """Phase 1 stub. Full implementation in Session 5+."""

    async def route(self, query: str) -> str:
        """Returns tier: 'hot', 'warm', or 'cold'."""
        return "warm"  # Default to Postgres
