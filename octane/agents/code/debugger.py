"""Debugger — analyzes stderr + broken code, asks LLM to produce a fix.

Called by CodeAgent when Validator says should_retry=True.
The fixed code is fed back into Executor for another attempt.
Max retries is enforced by CodeAgent (not here).
"""

from __future__ import annotations

import re
import structlog

from octane.tools.bodega_inference import BodegaInferenceClient

logger = structlog.get_logger().bind(component="code.debugger")

_SYSTEM_PROMPT = """\
You are an expert Python debugger. You will be given Python code that failed with an error.
Fix the code so it runs correctly.
Rules:
- Output ONLY the corrected Python source code, no markdown fences, no explanation
- Make the minimal change needed to fix the error
- Do not change the overall approach, just fix the bug"""


class Debugger:
    """Analyzes a failed execution and rewrites the code to fix the error."""

    def __init__(self, bodega: BodegaInferenceClient | None = None) -> None:
        self._bodega = bodega or BodegaInferenceClient()

    async def debug(self, code: str, error_summary: str) -> str:
        """Return fixed code. Falls back to original code if LLM is unavailable."""
        fixed = await self._fix_with_llm(code, error_summary)
        if fixed:
            logger.info("debug_fix_applied", error_len=len(error_summary))
            return fixed
        logger.warning("debug_llm_failed_returning_original")
        return code  # Can't fix without LLM — let retry count exhaust

    async def _fix_with_llm(self, code: str, error_summary: str) -> str | None:
        try:
            user_msg = (
                f"Error:\n{error_summary}\n\n"
                f"Code:\n```python\n{code}\n```"
            )
            raw = await self._bodega.chat_simple(
                prompt=user_msg,
                system=_SYSTEM_PROMPT,
                temperature=0.1,
            )
            fixed = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
            fixed = re.sub(r"\n?```$", "", fixed.strip(), flags=re.MULTILINE)
            return fixed if fixed.strip() else None
        except Exception as e:
            logger.warning("debug_llm_error", error=str(e))
            return None
