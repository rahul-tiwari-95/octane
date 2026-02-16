"""Writer — specification → actual code generation."""

from __future__ import annotations


class Writer:
    """Phase 1 stub. Full implementation in Session 4 (Code Agent)."""

    async def write(self, spec: dict) -> str:
        return f"# Stub code for: {spec.get('approach', 'unknown')}\nprint('hello')"
