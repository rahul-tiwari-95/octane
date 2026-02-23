"""Validator — checks execution result, decides pass/fail/retry.

Rules:
  - exit_code == 0 AND stdout non-empty → pass
  - exit_code == 0 AND stdout empty     → warn (might be silent success)
  - exit_code != 0                      → fail, extract error summary
  - Timeout (exit_code == 124)          → fail with timeout message
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger().bind(component="code.validator")


class Validator:
    """Validates execution output and decides whether to retry."""

    def validate(self, exec_result: dict) -> dict:
        """
        Returns:
            {
                "passed": bool,
                "should_retry": bool,
                "error_summary": str | None,
                "output": str,   # the clean stdout to show the user
            }
        """
        exit_code = exec_result.get("exit_code", 1)
        stdout = exec_result.get("stdout", "").strip()
        stderr = exec_result.get("stderr", "").strip()

        if exit_code == 124:
            logger.warning("execution_timeout")
            return {
                "passed": False,
                "should_retry": False,  # Timeout → no point retrying same code
                "error_summary": "Execution timed out.",
                "output": "",
            }

        if exit_code == 0:
            if not stdout:
                logger.warning("silent_success")
            else:
                logger.info("validation_passed")
            return {
                "passed": True,
                "should_retry": False,
                "error_summary": None,
                "output": stdout,
            }

        # Execution failed — extract useful error summary from stderr
        error_summary = self._extract_error(stderr)
        logger.warning("validation_failed", exit_code=exit_code, error=error_summary[:120])
        return {
            "passed": False,
            "should_retry": True,
            "error_summary": error_summary,
            "output": stdout,
        }

    def _extract_error(self, stderr: str) -> str:
        """Pull out the most useful part of a Python traceback."""
        if not stderr:
            return "Unknown error (no stderr)"
        lines = stderr.strip().splitlines()
        # Last 8 lines typically contain the actual error
        tail = "\n".join(lines[-8:]) if len(lines) > 8 else stderr
        return tail
