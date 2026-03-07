"""ChainExecutor — executes a parsed chain of steps in order.

The ChainExecutor runs a ChainPlan produced by ChainParser.  Each step is
executed sequentially, with the output of prior steps available as {refs}
for subsequent steps.

Execution model:
    1. For each step in order:
       a. Interpolate {prev}, {step_name}, {all}, {{var}} references.
       b. Dispatch the command to the appropriate agent/pipeline.
       c. Store the output under the step's name.
       d. Emit a progress event.
    2. Return ChainResult with all step outputs.

Command dispatch:
    "ask"        → OSA Orchestrator.run(interpolated_args)
    "search"     → WebAgent.search(interpolated_args)
    "fetch"      → WebAgent.fetch(interpolated_args)
    "analyze"    → WebAgent.ask with sub_agent=finance
    "synthesize" → Evaluator.synthesize(all outputs so far)
    "code"       → CodeAgent.run(interpolated_args)
    other        → Falls back to OSA Orchestrator.run() (generic routing)

--save support:
    If save_path is provided, the chain definition is serialized to JSON
    so it can be re-run later as `octane workflow run <file>`.

Progress events streamed:
    {"type": "step_start",  "data": {"index": int, "name": str, "command": str}}
    {"type": "step_done",   "data": {"name": str, "output": str, "latency_ms": float}}
    {"type": "step_error",  "data": {"name": str, "error": str}}
    {"type": "done",        "data": {"n_steps": int, "n_successful": int, "total_ms": float}}
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from octane.osa.chain_parser import ChainPlan, ChainStep, ChainParser

logger = structlog.get_logger().bind(component="osa.chain_executor")

_UNSET = object()


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass
class StepResult:
    """Result of one executed chain step.

    Attributes:
        step:       The ChainStep that was executed.
        output:     The text output of the step.
        latency_ms: Time taken to execute the step.
        error:      Non-empty if the step failed.
    """

    step: ChainStep
    output: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @property
    def success(self) -> bool:
        return bool(self.output) and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.step.index,
            "name": self.step.name,
            "command": self.step.command,
            "output": self.output,
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ChainResult:
    """Full output of a chain execution.

    Attributes:
        plan:       The ChainPlan that was executed.
        results:    Per-step results.
        total_ms:   End-to-end latency.
        saved_to:   Path if --save was used.
    """

    plan: ChainPlan
    results: list[StepResult] = field(default_factory=list)
    total_ms: float = 0.0
    saved_to: str | None = None

    @property
    def outputs(self) -> dict[str, str]:
        """Map step name → output for all completed steps."""
        return {r.step.name: r.output for r in self.results if r.success}

    @property
    def final_output(self) -> str:
        """The output of the last successful step."""
        for r in reversed(self.results):
            if r.success:
                return r.output
        return ""

    @property
    def successful_steps(self) -> list[StepResult]:
        return [r for r in self.results if r.success]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": len(self.results),
            "n_successful": len(self.successful_steps),
            "total_ms": round(self.total_ms, 1),
            "saved_to": self.saved_to,
            "results": [r.to_dict() for r in self.results],
            "final_output": self.final_output,
        }


# ── ChainExecutor ─────────────────────────────────────────────────────────────


class ChainExecutor:
    """Executes a ChainPlan step by step.

    Args:
        bodega:       BodegaRouter for LLM calls.
        web_agent:    WebAgent for search/fetch/analyze steps.
        orchestrator: OSA Orchestrator for 'ask' and generic dispatch.
        template_vars: Runtime variable values for {{var}} substitution.
    """

    def __init__(
        self,
        bodega=_UNSET,
        web_agent=None,
        orchestrator=None,
        template_vars: dict[str, str] | None = None,
    ) -> None:
        if bodega is _UNSET:
            from octane.tools.bodega_router import BodegaRouter
            bodega = BodegaRouter()
        self._bodega = bodega
        self._web_agent = web_agent
        self._orchestrator = orchestrator
        self.template_vars = template_vars or {}

    async def run_stream(
        self,
        plan: ChainPlan,
        session_id: str = "cli",
        save_path: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Execute a chain plan and yield progress events.

        Args:
            plan:       Parsed chain plan from ChainParser.
            session_id: For tracing.
            save_path:  If provided, save chain definition to this path.

        Yields:
            step_start → step_done|step_error → done
        """
        t0 = time.monotonic()
        step_outputs: dict[str, str] = {}  # name → output (accumulated)
        results: list[StepResult] = []

        for step in plan.steps:
            yield {
                "type": "step_start",
                "data": {
                    "index": step.index,
                    "name": step.name,
                    "command": step.command,
                    "args_template": step.args,
                },
            }

            # Interpolate references — replace {prev}, {step_name}, {all}, {{var}}
            resolved_args = step.interpolate(
                step_outputs=step_outputs,
                template_vars=self.template_vars,
            )

            step_t0 = time.monotonic()
            output = ""
            error = ""

            try:
                output = await self._dispatch(
                    command=step.command,
                    args=resolved_args,
                    session_id=session_id,
                )
            except Exception as exc:
                error = str(exc)
                logger.warning(
                    "chain_step_failed",
                    step=step.name,
                    command=step.command,
                    error=error,
                )

            latency_ms = (time.monotonic() - step_t0) * 1000
            result = StepResult(
                step=step,
                output=output,
                latency_ms=latency_ms,
                error=error,
            )
            results.append(result)

            # Store output for subsequent reference interpolation
            if output:
                step_outputs[step.name] = output
            # Also store with index key so {step_N} works even without a name
            if output:
                step_outputs[f"step_{step.index + 1}"] = output

            if error:
                yield {"type": "step_error", "data": result.to_dict()}
            else:
                yield {"type": "step_done", "data": result.to_dict()}

        total_ms = (time.monotonic() - t0) * 1000
        n_successful = sum(1 for r in results if r.success)

        saved_to: str | None = None
        if save_path:
            saved_to = self._save_chain(plan, save_path)

        yield {
            "type": "done",
            "data": {
                "n_steps": len(results),
                "n_successful": n_successful,
                "total_ms": round(total_ms, 1),
                "saved_to": saved_to,
                "final_output": results[-1].output if results else "",
            },
        }

    async def run(
        self,
        plan: ChainPlan,
        session_id: str = "cli",
        save_path: str | None = None,
    ) -> ChainResult:
        """Run a chain and return a complete ChainResult."""
        t0 = time.monotonic()
        results: list[StepResult] = []
        saved_to: str | None = None

        async for event in self.run_stream(plan, session_id=session_id, save_path=save_path):
            if event["type"] in ("step_done", "step_error"):
                d = event["data"]
                # Reconstruct StepResult from dict (already captured above in run_stream)
                # For run(), we just use the data directly
                pass
            elif event["type"] == "done":
                saved_to = event["data"].get("saved_to")

        # Re-run via a tracking async generator to capture StepResults directly
        t0 = time.monotonic()
        step_outputs: dict[str, str] = {}
        results = []

        for step in plan.steps:
            resolved_args = step.interpolate(
                step_outputs=step_outputs,
                template_vars=self.template_vars,
            )
            step_t0 = time.monotonic()
            output = ""
            error = ""
            try:
                output = await self._dispatch(step.command, resolved_args, session_id)
            except Exception as exc:
                error = str(exc)

            latency_ms = (time.monotonic() - step_t0) * 1000
            result = StepResult(step=step, output=output, latency_ms=latency_ms, error=error)
            results.append(result)
            if output:
                step_outputs[step.name] = output
                step_outputs[f"step_{step.index + 1}"] = output

        if save_path:
            saved_to = self._save_chain(plan, save_path)

        return ChainResult(
            plan=plan,
            results=results,
            total_ms=(time.monotonic() - t0) * 1000,
            saved_to=saved_to,
        )

    async def _dispatch(
        self,
        command: str,
        args: str,
        session_id: str,
    ) -> str:
        """Dispatch a single step to the appropriate backend.

        Routes:
            ask / (default) → OSA Orchestrator or BodegaRouter direct
            search / fetch  → WebAgent
            analyze         → WebAgent (finance mode)
            synthesize      → Evaluator/BodegaRouter
            code            → CodeAgent (if available) or stub
        """
        args = args.strip()

        if command in ("ask",) or command not in (
            "search", "fetch", "analyze", "synthesize", "code"
        ):
            # Route through orchestrator if available
            if self._orchestrator is not None:
                try:
                    result = await self._orchestrator.run(
                        args,
                        session_id=session_id,
                    )
                    raw = result.output or str(result.data) if result else ""
                    return _strip_think(raw)
                except Exception as exc:
                    logger.warning("chain_orchestrator_failed", error=str(exc))

            # Direct bodega fallback
            if self._bodega:
                try:
                    raw = await self._bodega.chat_simple(
                        args,
                        tier=__import__(
                            "octane.tools.topology", fromlist=["ModelTier"]
                        ).ModelTier.MID,
                        max_tokens=1500,
                    )
                    return _strip_think(raw)
                except Exception as exc:
                    raise RuntimeError(f"Bodega dispatch failed: {exc}") from exc
            return f"[No executor available for: {command} {args}]"

        elif command in ("search", "fetch"):
            if self._web_agent is not None:
                try:
                    from octane.models.schemas import AgentRequest
                    sub = "finance" if "finance" in args.lower() else "search"
                    request = AgentRequest(
                        query=args,
                        source="chain",
                        session_id=session_id,
                        metadata={"sub_agent": sub},
                    )
                    response = await self._web_agent.execute(request)
                    return response.output or str(response.data) if response else ""
                except Exception as exc:
                    raise RuntimeError(f"Web agent failed: {exc}") from exc
            return f"[Web agent unavailable for: {args}]"

        elif command == "analyze":
            if self._web_agent is not None:
                try:
                    from octane.models.schemas import AgentRequest
                    request = AgentRequest(
                        query=args,
                        source="chain",
                        session_id=session_id,
                        metadata={"sub_agent": "finance"},
                    )
                    response = await self._web_agent.execute(request)
                    return response.output or str(response.data) if response else ""
                except Exception as exc:
                    raise RuntimeError(f"Analysis failed: {exc}") from exc
            return f"[Analysis agent unavailable for: {args}]"

        elif command == "synthesize":
            if self._bodega:
                from octane.tools.topology import ModelTier
                prompt = f"Synthesize the following into a clear summary:\n\n{args}"
                # Try REASON tier first (8B), fall back to INSTRUCT (loaded model)
                for tier in (ModelTier.REASON, ModelTier.MID, ModelTier.INSTRUCT):
                    try:
                        raw = await self._bodega.chat_simple(
                            prompt,
                            tier=tier,
                            max_tokens=2000,
                            temperature=0.1,
                        )
                        return _strip_think(raw)
                    except Exception as exc:
                        logger.warning(
                            "chain_synthesize_tier_failed",
                            tier=tier.value if hasattr(tier, "value") else str(tier),
                            error=str(exc),
                        )
                raise RuntimeError("Synthesis failed: no available model tier responded")
            return args  # Pass-through if no bodega

        elif command == "code":
            # Code execution — stub for now, full CodeAgent hookup in future
            return f"[Code execution not yet wired in chain: {args}]"

        return f"[Unknown command: {command}]"

    @staticmethod
    def _save_chain(plan: ChainPlan, save_path: str) -> str:
        """Save the chain definition to a JSON file.

        Returns the resolved save path.
        """
        path = Path(save_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "type": "octane_chain",
            "version": "1.0",
            "steps": [
                {"name": s.name, "command": s.command, "args": s.args}
                for s in plan.steps
            ],
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("chain_saved", path=str(path))
        return str(path)


# ── Convenience: parse + run ──────────────────────────────────────────────────


async def run_chain(
    step_strings: list[str],
    template_vars: dict[str, str] | None = None,
    bodega=_UNSET,
    web_agent=None,
    orchestrator=None,
    session_id: str = "cli",
    save_path: str | None = None,
) -> ChainResult:
    """Parse and execute a chain in one call.

    Convenience function for the CLI — combines ChainParser + ChainExecutor.

    Args:
        step_strings:  Raw step strings from the CLI.
        template_vars: Runtime {{var}} values.
        bodega:        BodegaRouter (or None for fallback).
        web_agent:     WebAgent (or None).
        orchestrator:  OSA Orchestrator (or None).
        session_id:    For tracing.
        save_path:     If provided, save chain definition to file.

    Returns:
        ChainResult with all step outputs.
    """
    parser = ChainParser()
    plan = parser.parse(step_strings, template_vars=template_vars)

    executor = ChainExecutor(
        bodega=bodega,
        web_agent=web_agent,
        orchestrator=orchestrator,
        template_vars=template_vars or {},
    )
    return await executor.run(plan, session_id=session_id, save_path=save_path)
