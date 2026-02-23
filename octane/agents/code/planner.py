"""Planner — task description → code specification.

Produces a structured spec dict that the Writer consumes:
    {
        "language": "python",
        "task": <original task>,
        "approach": <brief description>,
        "steps": [<step1>, <step2>, ...],
        "requirements": [<pkg1>, ...],   # pip packages needed
        "entry_point": "solution.py",
    }

If Bodega Inference is offline the spec is produced by keyword heuristics.
"""

from __future__ import annotations

import re
import structlog

from octane.tools.bodega_inference import BodegaInferenceClient

logger = structlog.get_logger().bind(component="code.planner")

_SYSTEM_PROMPT = """\
You are a senior software engineer. Given a coding task, output a JSON planning spec with exactly these keys:
- language: programming language (usually "python")
- approach: one-sentence description of the solution strategy
- steps: list of 3-6 implementation steps as plain strings
- requirements: list of pip packages needed (empty list if none beyond stdlib)
- entry_point: filename to write the solution to (e.g. "solution.py")

Respond with ONLY valid JSON. No markdown fences, no prose."""


class Planner:
    """Converts a natural language coding task into a structured spec."""

    def __init__(self, bodega: BodegaInferenceClient | None = None) -> None:
        self._bodega = bodega or BodegaInferenceClient()

    async def plan(self, task: str) -> dict:
        """Return a spec dict for the given task."""
        spec = await self._plan_with_llm(task)
        if spec:
            logger.info("plan_llm", task=task[:60])
            return spec
        logger.warning("plan_llm_failed_using_heuristic", task=task[:60])
        return self._heuristic_plan(task)

    async def _plan_with_llm(self, task: str) -> dict | None:
        import json
        try:
            raw = await self._bodega.chat_simple(
                prompt=f"Task: {task}",
                system=_SYSTEM_PROMPT,
                temperature=0.2,
            )
            # Strip any accidental markdown fences
            raw = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
            raw = re.sub(r"\n?```$", "", raw.strip(), flags=re.MULTILINE)
            return json.loads(raw)
        except Exception as e:
            logger.warning("plan_llm_error", error=str(e))
            return None

    def _heuristic_plan(self, task: str) -> dict:
        """Minimal deterministic spec when LLM is unavailable."""
        task_lower = task.lower()
        reqs: list[str] = []
        if any(w in task_lower for w in ["plot", "chart", "graph", "visuali"]):
            reqs.append("matplotlib")
        if any(w in task_lower for w in ["dataframe", "csv", "pandas"]):
            reqs.append("pandas")
        if any(w in task_lower for w in ["request", "http", "fetch", "api"]):
            reqs.append("httpx")
        if any(w in task_lower for w in ["numpy", "array", "matrix", "np."]):
            reqs.append("numpy")
        return {
            "language": "python",
            "task": task,
            "approach": f"Implement: {task}",
            "steps": [
                "Define the problem and inputs",
                "Implement core logic",
                "Print or return the result",
            ],
            "requirements": reqs,
            "entry_point": "solution.py",
        }
