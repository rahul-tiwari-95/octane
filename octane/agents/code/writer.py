"""Writer — spec dict → runnable Python code.

Uses Bodega Inference (LLM) to generate the code. Falls back to a
minimal template stub when the LLM is offline so the pipeline never
hard-crashes.
"""

from __future__ import annotations

import re
import structlog

from octane.tools.bodega_inference import BodegaInferenceClient

logger = structlog.get_logger().bind(component="code.writer")

_SYSTEM_PROMPT = """\
You are an expert Python developer. Write clean, complete, runnable Python code for the given task spec.
Rules:
- Output ONLY the Python source code, no markdown fences, no explanation
- The code must be self-contained and runnable with `python solution.py`
- Import only packages listed in requirements (plus stdlib)
- Print the final answer or result to stdout
- The script MUST complete in under 5 seconds. Use iterative algorithms, never recursion for sequences.
- No infinite loops. No sleep(). No blocking I/O unless explicitly asked.
- Handle errors gracefully with try/except where appropriate
- Keep it concise: solve the problem, don't over-engineer"""


class Writer:
    """Generates runnable Python code from a planning spec."""

    def __init__(self, bodega: BodegaInferenceClient | None = None) -> None:
        self._bodega = bodega or BodegaInferenceClient()

    async def write(self, spec: dict, previous_error: str | None = None) -> str:
        """Generate code from spec. If previous_error is set, this is a retry/debug pass."""
        code = await self._write_with_llm(spec, previous_error)
        if code:
            logger.info("code_written", approach=spec.get("approach", "")[:60])
            return code
        logger.warning("write_llm_failed_using_template")
        return self._template_fallback(spec)

    async def _write_with_llm(self, spec: dict, previous_error: str | None) -> str | None:
        try:
            user_parts = [
                f"Task: {spec.get('task', spec.get('approach', 'unknown'))}",
                f"Approach: {spec.get('approach', '')}",
                f"Steps: {', '.join(spec.get('steps', []))}",
            ]
            if spec.get("requirements"):
                user_parts.append(f"Available packages: {', '.join(spec['requirements'])}")
            if previous_error:
                user_parts.append(
                    f"\nPrevious attempt FAILED with this error:\n{previous_error}\n"
                    "Fix the error and write corrected code."
                )
            raw = await self._bodega.chat_simple(
                prompt="\n".join(user_parts),
                system=_SYSTEM_PROMPT,
                temperature=0.2,
            )
            # Strip accidental markdown fences
            code = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
            code = re.sub(r"\n?```$", "", code.strip(), flags=re.MULTILINE)
            if not code.strip():
                return None
            return code
        except Exception as e:
            logger.warning("write_llm_error", error=str(e))
            return None

    def _template_fallback(self, spec: dict) -> str:
        task = spec.get("task", spec.get("approach", "unknown task"))
        reqs = spec.get("requirements", [])
        imports = "\n".join(f"import {r}" for r in reqs) if reqs else ""
        return f'''\
#!/usr/bin/env python3
"""Auto-generated: {task}"""
{imports}

def main():
    # TODO: implement solution for: {task}
    print("Solution for: {task}")
    print("(LLM offline — this is a placeholder)")

if __name__ == "__main__":
    main()
'''
