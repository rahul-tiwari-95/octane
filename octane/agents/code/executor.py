"""Executor — venv creation, pip install, run code, capture output."""

from __future__ import annotations


class Executor:
    """Phase 1 stub. Full implementation in Session 4 (Code Agent)."""

    async def run(self, code: str, language: str = "python") -> dict[str, str]:
        return {"stdout": "", "stderr": "Executor stub", "exit_code": "1"}
