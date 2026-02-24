"""Memory Router — classifies intent as READ or WRITE.

READ:  "what was ...", "do you remember ...", "recall ...", "last time ..."
WRITE: everything else that produced a meaningful answer worth saving

The router also generates the Redis key for the operation.
Key scheme:  memory:{session_id}:{slot}
"""

from __future__ import annotations

import re
import structlog

logger = structlog.get_logger().bind(component="memory.router")

_READ_PATTERNS = re.compile(
    r"\b(remember|recall|last|previous|earlier|before|what was|what did|you said|"
    r"told me|mentioned|stored|saved)\b",
    re.IGNORECASE,
)


class MemoryRouter:
    """Decides whether an operation is a READ or WRITE and builds the cache key."""

    def route(self, query: str, session_id: str = "default") -> dict:
        """
        Returns:
            {
                "intent": "read" | "write",
                "session_id": str,
                "slot": str,   # short label for the key
                "key": str,    # full Redis key
            }
        """
        intent = "read" if _READ_PATTERNS.search(query) else "write"
        slot = self._extract_slot(query)
        key = f"memory:{session_id}:{slot}"
        logger.debug("memory_route", intent=intent, key=key)
        return {"intent": intent, "session_id": session_id, "slot": slot, "key": key}

    def _extract_slot(self, query: str) -> str:
        """Generate a short, stable slot label from the query text."""
        # Strip common stop words, take first 3 meaningful words
        stopwords = {"what", "was", "were", "is", "are", "the", "a", "an",
                     "of", "for", "in", "on", "at", "to", "and", "or",
                     "do", "did", "you", "me", "my", "i", "tell", "show",
                     "give", "get", "has", "had", "have", "been", "be"}
        words = re.findall(r"[a-z0-9]+", query.lower())
        meaningful = [w for w in words if w not in stopwords][:3]
        return "_".join(meaningful) if meaningful else "general"
