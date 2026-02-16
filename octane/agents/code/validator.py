"""Validator — output verification."""

from __future__ import annotations


class Validator:
    """Phase 1 stub. Full implementation in Session 4 (Code Agent)."""

    async def validate(self, output: dict[str, str]) -> bool:
        return output.get("exit_code") == "0"
