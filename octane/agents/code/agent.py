"""Code Agent coordinator.

Pipeline: Planner -> Writer -> Executor -> Validator -> (Debugger -> Writer -> Executor) x N

Before entering the LLM pipeline, CodeAgent checks the CatalystRegistry for a
deterministic pre-written solution. If a catalyst matches the query + upstream data,
it runs directly and returns -- no LLM call, no sandbox execution needed.

Self-healing loop (LLM path):
    1. Plan the task
    2. Write initial code
    3. Execute in isolated subprocess
    4. Validate output
    5. If failed and retries remain: Debugger rewrites -> back to step 3
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
from octane.catalysts.registry import CatalystRegistry
from octane.models.schemas import AgentRequest, AgentResponse
from octane.models.synapse import SynapseEvent
from octane.tools.bodega_inference import BodegaInferenceClient
from octane.tools.structured_store import ArtifactStore

logger = structlog.get_logger().bind(component="code.agent")

MAX_RETRIES = 3


class CodeAgent(BaseAgent):
    """Code Agent -- generates, executes, and self-heals Python code."""

    name = "code"

    def __init__(self, synapse, bodega: BodegaInferenceClient | None = None,
                 artifact_store: ArtifactStore | None = None) -> None:
        super().__init__(synapse)
        bodega = bodega or BodegaInferenceClient()
        self.planner = Planner(bodega=bodega)
        self.writer = Writer(bodega=bodega)
        self.executor = Executor()
        self.validator = Validator()
        self.debugger = Debugger(bodega=bodega)
        self.catalyst_registry = CatalystRegistry()
        self._artifact_store = artifact_store

    async def execute(self, request: AgentRequest) -> AgentResponse:
        """Run the plan -> write -> execute -> validate -> debug loop.

        Checks CatalystRegistry first. If a catalyst matches, returns immediately
        without touching the LLM pipeline or sandbox executor.
        """
        task = request.query
        upstream_results: dict = request.context.get("upstream_results", {})

        # -- CATALYST CHECK ----------------------------------------------------
        catalyst_match = self.catalyst_registry.match(task, upstream_results)
        if catalyst_match:
            return await self._run_catalyst(catalyst_match, task, request)

        # -- LLM PIPELINE ------------------------------------------------------
        return await self._run_llm_pipeline(task, request)

    async def _run_catalyst(
        self,
        catalyst_match: tuple,
        task: str,
        request: AgentRequest,
    ) -> AgentResponse:
        """Execute a matched catalyst function and wrap result as AgentResponse."""
        catalyst_fn, resolved_data = catalyst_match
        output_dir = self.executor.get_output_dir(request.correlation_id)

        try:
            result = catalyst_fn(
                resolved_data,
                output_dir,
                correlation_id=request.correlation_id,
                instruction=task,
            )
            summary = result.get("summary", "Catalyst completed successfully.")
            artifacts = result.get("artifacts", [])

            self.synapse.emit(SynapseEvent(
                correlation_id=request.correlation_id or "unknown",
                event_type="catalyst_success",
                source="code.agent",
                payload={
                    "catalyst": catalyst_fn.__name__,
                    "artifacts": artifacts,
                    "output_dir": output_dir,
                },
            ))
            logger.info(
                "catalyst_success",
                catalyst=catalyst_fn.__name__,
                artifacts=len(artifacts),
                output_dir=output_dir,
            )

            output_lines = [f"Catalyst: {catalyst_fn.__name__}", "", summary]
            if artifacts:
                output_lines += ["", "Saved files:"] + [f"  {a}" for a in artifacts]

            return AgentResponse(
                agent=self.name,
                success=True,
                output="\n".join(output_lines),
                data=result,
                correlation_id=request.correlation_id,
                metadata={"via_catalyst": catalyst_fn.__name__},
            )

        except Exception as exc:
            # Catalyst failed -- log loudly and fall through to LLM pipeline
            logger.warning(
                "catalyst_failed_falling_through",
                catalyst=catalyst_fn.__name__,
                error=str(exc),
            )
            self.synapse.emit(SynapseEvent(
                correlation_id=request.correlation_id or "unknown",
                event_type="catalyst_failed",
                source="code.agent",
                payload={"catalyst": catalyst_fn.__name__, "error": str(exc)},
            ))
            return await self._run_llm_pipeline(task, request)

    async def _run_llm_pipeline(self, task: str, request: AgentRequest) -> AgentResponse:
        """Full LLM code-gen pipeline (Planner -> Writer -> Executor -> Validator -> Debugger)."""
        spec = await self.planner.plan(task)
        logger.info("plan_ready", approach=spec.get("approach", "")[:80])

        code = await self.writer.write(spec)
        requirements = spec.get("requirements", [])

        last_error: str | None = None
        exec_result: dict = {}
        validation: dict = {}

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info("executing_attempt", attempt=attempt)

            exec_result = await self.executor.run(
                code,
                requirements=requirements,
                correlation_id=request.correlation_id,
            )
            validation = self.validator.validate(exec_result)

            if validation["passed"]:
                logger.info("code_success", attempt=attempt)
                self.synapse.emit(SynapseEvent(
                    correlation_id=request.correlation_id or "unknown",
                    event_type="code_healed" if attempt > 1 else "code_success",
                    source="code.agent",
                    payload={
                        "attempt": attempt,
                        "duration_ms": exec_result.get("duration_ms", 0),
                        "output_files": exec_result.get("output_files", []),
                        "healed": attempt > 1,
                    },
                ))
                break

            last_error = validation["error_summary"]
            logger.warning("code_failed", attempt=attempt, error=last_error[:120])

            self.synapse.emit(SynapseEvent(
                correlation_id=request.correlation_id or "unknown",
                event_type="code_attempt_failed",
                source="code.agent",
                payload={
                    "attempt": attempt,
                    "error_summary": last_error[:200],
                    "should_retry": validation["should_retry"],
                    "exit_code": exec_result.get("exit_code"),
                },
            ))

            if not validation["should_retry"] or attempt == MAX_RETRIES:
                break

            logger.info("invoking_debugger", attempt=attempt)
            self.synapse.emit(SynapseEvent(
                correlation_id=request.correlation_id or "unknown",
                event_type="code_debug_invoked",
                source="code.agent",
                payload={"attempt": attempt, "error_len": len(last_error)},
            ))
            fixed_code = await self.debugger.debug(code, last_error)
            if fixed_code and fixed_code != code:
                code = fixed_code
                logger.info("code_rewritten_by_debugger", attempt=attempt)
            else:
                logger.warning("debugger_no_change_stopping", attempt=attempt)
                break

        if validation.get("passed"):
            output_text = self._format_success(task, spec, code, exec_result, validation)
            # Persist generated artifact (fire-and-forget, non-blocking)
            if self._artifact_store is not None:
                try:
                    await self._artifact_store.register(
                        content=code,
                        artifact_type="code",
                        language=spec.get("language", "python"),
                        description=task[:200],
                        session_id=request.correlation_id or "",
                    )
                except Exception as _ae:
                    logger.warning("artifact_store_error", error=str(_ae))
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
            self.synapse.emit(SynapseEvent(
                correlation_id=request.correlation_id or "unknown",
                event_type="code_exhausted",
                source="code.agent",
                payload={
                    "attempts": MAX_RETRIES,
                    "last_error": (last_error or "")[:200],
                    "task": task[:120],
                },
            ))
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
        artifacts = exec_result.get("output_files", [])
        output_dir = exec_result.get("output_dir")

        lines = [
            f"Code executed successfully ({duration:.0f}ms)",
            f"Task: {task}",
        ]
        if artifacts and output_dir:
            lines.append("")
            lines.append("Saved files:")
            for f in artifacts:
                lines.append(f"  {output_dir}/{f}")
        lines += ["", "Output:", output or "(no stdout)", "", "Code:", "```python", code.strip(), "```"]
        return "\n".join(lines)

    def _format_failure(self, task: str, error: str | None, code: str) -> str:
        lines = [
            f"Code execution failed after {MAX_RETRIES} attempts",
            f"Task: {task}",
            "",
            "Last Error:",
            error or "Unknown error",
            "",
            "Last Code Attempt:",
            "```python",
            code.strip(),
            "```",
        ]
        return "\n".join(lines)
