"""Writer — spec dict → runnable Python code.

Uses Bodega Inference (LLM) to generate the code. Falls back to a
minimal template stub when the LLM is offline so the pipeline never
hard-crashes.
"""

from __future__ import annotations

import re
import structlog

from octane.tools.bodega_inference import BodegaInferenceClient
from octane.tools.topology import ModelTier

logger = structlog.get_logger().bind(component="code.writer")

# Regex that matches lines that look like Python code (rough heuristic)
_PY_LINE_RE = re.compile(
    r"^\s*(import |from |def |class |#|@|if |else:|elif |for |while |try:|except|"
    r"with |return |raise |yield |async |await |print\(|[a-zA-Z_][a-zA-Z0-9_.]*\s*[=(+\-\[{])"
)


def _strip_prose_lines(code: str) -> str:
    """Remove leading/trailing prose lines that aren't Python.

    Prose lines: plain English sentences with no Python syntax markers.
    We keep a line if it matches _PY_LINE_RE OR if it's blank OR if we're
    already inside the code body (once first real code line is seen, stop
    stripping from the front; strip only from the tail symmetrically).
    """
    lines = code.splitlines()
    # Strip leading prose
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or _PY_LINE_RE.match(line):
            start = i
            break
    # Strip trailing prose (walk backwards)
    end = len(lines)
    for i in range(len(lines) - 1, start - 1, -1):
        stripped = lines[i].strip()
        if not stripped or _PY_LINE_RE.match(lines[i]) or stripped.startswith((")", "]", "}")):
            end = i + 1
            break
    return "\n".join(lines[start:end])

_SYSTEM_PROMPT = """\
You are an expert Python developer. Write clean, complete, runnable Python code for the given task spec.
Rules:
- Output ONLY the Python source code, no markdown fences, no explanation, no prose
- The code must be self-contained and runnable with `python solution.py`
- Import only packages listed in requirements (plus stdlib)
- Print the final answer or result to stdout
- The script MUST complete in under 60 seconds.
- No infinite loops. No sleep(). No blocking I/O unless explicitly asked.
- Handle errors gracefully with try/except where appropriate
- Keep it concise: solve the problem, don't over-engineer
- For charts/plots: the variable OUTPUT_DIR is pre-defined — save the figure with
  plt.savefig(os.path.join(OUTPUT_DIR, 'chart.png'), dpi=150, bbox_inches='tight')
  then call plt.close(). NEVER call plt.show() — it blocks the process.
- When upstream data is provided as CSV, parse and use it directly. Do NOT
  re-fetch data from the internet unless no data was provided."""


class Writer:
    """Generates runnable Python code from a planning spec."""

    def __init__(self, bodega=None) -> None:
        if bodega is None:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega

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
                tier=ModelTier.REASON,
                temperature=0.2,
            )
            # Preserve <think> reasoning as a debug trace; extract code from after </think>
            if "</think>" in raw:
                think_part, _, code = raw.partition("</think>")
                reasoning_trace = think_part.replace("<think>", "").strip()
                logger.debug("model_reasoning_trace", trace=reasoning_trace[:500])
            else:
                code = raw
            # Strip accidental markdown fences
            code = re.sub(r"^```[a-z]*\n?", "", code.strip(), flags=re.MULTILINE)
            code = re.sub(r"\n?```$", "", code.strip(), flags=re.MULTILINE)
            # Strip any leading/trailing prose lines that slipped through
            # (lines that don't look like Python: no =, no import, no def, no #, no indent)
            code = _strip_prose_lines(code)
            if not code.strip():
                return None
            return code.strip()
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
