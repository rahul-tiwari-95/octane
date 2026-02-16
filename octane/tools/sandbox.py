"""Sandbox — venv creation + subprocess execution for Code Agent (stub).

Full implementation in Session 4 when Code Agent is built.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="sandbox")


class Sandbox:
    """Isolated code execution environment.

    Phase 1: Stub.
    Phase 4 (Code Agent): Full venv creation, pip install, subprocess execution.
    """

    async def execute(self, code: str, language: str = "python") -> dict[str, str]:
        """Execute code in a sandboxed environment (stub)."""
        logger.warning("sandbox_stub", msg="Sandbox is a stub in Phase 1")
        return {"stdout": "", "stderr": "Sandbox not implemented yet", "exit_code": "1"}
