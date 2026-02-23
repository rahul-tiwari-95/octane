"""Code Agent coordinator.

Pipeline: Planner → Writer → Executor → Validator → (Debugger → Writer → Executor) × N

Self-healing loop:
    1. Plan the task
    2. Write initial code
    3. Execute in isolated subprocess
    4. Validate output
    5. If failed and retries remain: Debugger rewrites → back to step 3
    6. Return best result (or final error if all retries exhausted)
"""

from __future__ import annotations

import structlog

from octane.agents.base import BaseAgent
from octane.agents.code.debugger import Debugger
from octane.agents.code.executor import Executor
from octane.agents.code.planner import Planner
from octane.agents.code.validator import Validator
from octane.agents.code.writer import Writer
from octane.models.schemas import AgentRequest, AgentResponse
from octane.tools.bodega_inference import BodegaInferenceClient

logger = structlog.get_logger().bind(component="code.agent")

MAX_RETRIES = 3


class CodeAgent(BaseAgent):
    """Code Agent — generates, executes, and self-heals Python code."""

    name = "code"

    def __init__(self, synapse, bodega: BodegaInferenceClient | None = None) -> None:
        super().__init__(synapse)
        bodega = bodega or BodegaInferenceClient()
        self.planner = Planner(bodega=bodega)
        self.writer = Writer(bodega=bodega)
        self.executor = Executor()
        self.validator = Validator()
        self.debugger = Debugger(bodega=bodega)

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Run the full plan → write → execute → validate → debug loop."""
        task = request.query

        # ── PLAN ──────────────────────────────────────────────────────────
        spec = await self.planner.plan(task)
        logger.info("plan_ready", approach=spec.get("approach", "")[:80])

        # ── WRITE ─────────────────────────────────────────────────────────
        code = await self.writer.write(spec)
        requirements = spec.get("requirements", [])

        last_error: str | None = None
        exec_result: dict = {}
        validation: dict = {}

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info("executing_attempt", attempt=attempt)

            # ── EXECUTE ───────────────────────────────────────────────────
            exec_result = await self.executor.run(code, requirements=requirements)

            # ── VALIDATE ──────────────────────────────────────────────────
            validation = self.validator.validate(exec_result)

            if validation["passed"]:
                logger.info("code_success", attempt=attempt)
                break

            last_error = validation["error_summary"]
            logger.warning("code_failed", attempt=attempt, error=last_error[:120])

            if not validation["should_retry"] or attempt == MAX_RETRIES:
                break

            # ── DEBUG → REWRITE ───────────────────────────────────────────
            logger.info("invoking_debugger", attempt=attempt)
            code = await self.debugger.debug(code, last_error)

        # ── FORMAT RESPONSE ───────────────────────────────────────────────
        if validation.get("passed"):
            output_text = self._format_success(task, spec, code, exec_result, validation)
            return AgentResponse(
                agent=self.name,
                success=True,
                output=output_text,
                data={
                    "code": code,
                    "stdout": exec_result.get("stdout", ""),
                    "spec": spec,
                },
                correlation_id=request.correlation_id,
            )
        else:
            output_text = self._format_failure(task, last_error, code)
            return AgentResponse(
                agent=self.name,
                success=False,
                output=output_text,
                error=last_error or "Code execution failed",
                data={
                    "code": code,
                    "stderr": exec_result.get("stderr", ""),
                    "spec": spec,
                },
                correlation_id=request.correlation_id,
            )

    def _format_success(
        self, task: str, spec: dict, code: str, exec_result: dict, validation: dict
    ) -> str:
        output = validation.get("output", "").strip()
        duration = exec_result.get("duration_ms", 0)
        lines = [
            f"✅ Code executed successfully ({duration:.0f}ms)",
            f"Task: {task}",
            "",
            "── Output ──",
            output or "(no stdout)",
            "",
            "── Code ──",
            "```python",
            code.strip(),
            "```",
        ]
        return "\n".join(lines)

    def _format_failure(self, task: str, error: str | None, code: str) -> str:
        lines = [
            f"❌ Code execution failed after {MAX_RETRIES} attempts",
            f"Task: {task}",
            "",
            "── Last Error ──",
            error or "Unknown error",
            "",
            "── Last Code Attempt ──",
            "```python",
            code.strip(),
            "```",
        ]
        return "\n".join(lines)
