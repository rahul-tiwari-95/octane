"""Session 25 — Power Commands: tests for
    - DimensionPlanner (parsing, fallback, min/max dims)
    - ComparisonPlanner (parsing, extract_items, matrix)
    - ChainParser (steps, refs, validation, interpolation)
    - ChainExecutor (dispatch, {prev}/{all}, --save, error recovery)
    - InvestigateOrchestrator (run, stream events, no-agent fallback)
    - CompareOrchestrator (run, cell_for, matrix, no-agent fallback)
    - CLI commands registered in app (investigate / compare / chain)

All tests are pure-unit — no network, no Bodega, no Redis.
"""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run a coroutine synchronously in tests."""
    return asyncio.run(coro)


async def _collect_events(gen) -> list[dict]:
    """Drain an async iterator of events."""
    events = []
    async for ev in gen:
        events.append(ev)
    return events


# ─────────────────────────────────────────────────────────────────────────────
# DimensionPlanner
# ─────────────────────────────────────────────────────────────────────────────

class TestDimensionPlannerParsing:
    """Tests for DimensionPlanner._parse_response()."""

    def _make_planner(self, bodega_response: str):
        """Return a DimensionPlanner with a mocked Bodega."""
        from octane.osa.dimension_planner import DimensionPlanner
        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(return_value=bodega_response)
        return DimensionPlanner(bodega=mock_bodega)

    def _valid_json(self, n: int = 4) -> str:
        dims = [
            {
                "id": f"dim_{i}",
                "label": f"Dimension {i}",
                "queries": [f"query {i}"],
                "priority": i,
                "rationale": f"Reason {i}",
            }
            for i in range(1, n + 1)
        ]
        return json.dumps({"dimensions": dims})

    def test_parse_clean_json(self):
        planner = self._make_planner(self._valid_json(4))
        plan = _run(planner.plan("What is NVDA worth?"))
        assert len(plan.dimensions) == 4
        assert plan.from_llm is True
        assert plan.query == "What is NVDA worth?"

    def test_parse_strips_markdown_fence(self):
        raw = f"```json\n{self._valid_json(3)}\n```"
        planner = self._make_planner(raw)
        plan = _run(planner.plan("test"))
        assert len(plan.dimensions) == 3

    def test_parse_strips_think_tags(self):
        raw = f"<think>Let me think...</think>\n{self._valid_json(3)}"
        planner = self._make_planner(raw)
        plan = _run(planner.plan("test"))
        assert len(plan.dimensions) == 3

    def test_parse_strips_think_and_fence(self):
        raw = f"<think>Hmm</think>\n```json\n{self._valid_json(2)}\n```"
        planner = self._make_planner(raw)
        plan = _run(planner.plan("test"))
        assert len(plan.dimensions) == 2

    def test_sorted_dimensions_by_priority(self):
        dims = [
            {"id": "d3", "label": "C", "queries": ["q"], "priority": 3, "rationale": "r"},
            {"id": "d1", "label": "A", "queries": ["q"], "priority": 1, "rationale": "r"},
            {"id": "d2", "label": "B", "queries": ["q"], "priority": 2, "rationale": "r"},
        ]
        planner = self._make_planner(json.dumps({"dimensions": dims}))
        plan = _run(planner.plan("test"))
        sorted_ids = [d.id for d in plan.sorted_dimensions]
        assert sorted_ids == ["d1", "d2", "d3"]

    def test_respects_max_dimensions_arg(self):
        planner = self._make_planner(self._valid_json(8))
        plan = _run(planner.plan("test", max_dimensions=3))
        assert len(plan.dimensions) <= 3

    def test_min_dimensions_fallback_on_empty(self):
        planner = self._make_planner('{"dimensions": []}')
        plan = _run(planner.plan("test"))
        # Should fall back to keyword plan with at least 2 dims
        assert len(plan.dimensions) >= 2
        assert plan.from_llm is False

    def test_dimension_has_required_fields(self):
        planner = self._make_planner(self._valid_json(4))
        plan = _run(planner.plan("test"))
        for d in plan.dimensions:
            assert d.id
            assert d.label
            assert isinstance(d.queries, list)
            assert len(d.queries) >= 1
            assert d.primary_query() == d.queries[0]

    def test_to_dict_round_trip(self):
        planner = self._make_planner(self._valid_json(3))
        plan = _run(planner.plan("testing round trip"))
        d = plan.to_dict()
        assert d["query"] == "testing round trip"
        assert len(d["dimensions"]) == 3
        assert "from_llm" in d


class TestDimensionPlannerFallback:
    """Tests for keyword-based fallback when Bodega is unavailable."""

    def _make_failing_planner(self):
        from octane.osa.dimension_planner import DimensionPlanner
        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(side_effect=RuntimeError("offline"))
        return DimensionPlanner(bodega=mock_bodega)

    def test_fallback_returns_plan(self):
        from octane.osa.dimension_planner import DimensionPlanner
        planner = DimensionPlanner(bodega=None)
        plan = _run(planner.plan("test"))
        assert len(plan.dimensions) >= 2
        assert plan.from_llm is False

    def test_fallback_on_exception(self):
        planner = self._make_failing_planner()
        plan = _run(planner.plan("semiconductor supply chains"))
        assert plan is not None
        assert len(plan.dimensions) >= 2

    def test_finance_keyword_dims(self):
        from octane.osa.dimension_planner import DimensionPlanner
        planner = DimensionPlanner(bodega=None)
        plan = _run(planner.plan("Is NVDA stock overvalued?"))
        ids = [d.id for d in plan.dimensions]
        # Should produce finance-flavored dimensions
        assert len(ids) >= 3

    def test_generic_keyword_dims(self):
        from octane.osa.dimension_planner import DimensionPlanner
        planner = DimensionPlanner(bodega=None)
        plan = _run(planner.plan("How does photosynthesis work?"))
        assert len(plan.dimensions) >= 2

    def test_fallback_dims_have_queries(self):
        from octane.osa.dimension_planner import DimensionPlanner
        planner = DimensionPlanner(bodega=None)
        plan = _run(planner.plan("climate change"))
        for d in plan.dimensions:
            assert len(d.queries) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# ComparisonPlanner
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractItemsFromQuery:
    """Tests for the extract_items_from_query free function."""

    def test_vs_pattern(self):
        from octane.osa.comparison_planner import extract_items_from_query
        items = extract_items_from_query("NVDA vs AMD")
        assert len(items) == 2
        assert "NVDA" in items
        assert "AMD" in items

    def test_versus_pattern(self):
        from octane.osa.comparison_planner import extract_items_from_query
        items = extract_items_from_query("React versus Vue")
        assert len(items) == 2

    def test_compare_and_pattern(self):
        from octane.osa.comparison_planner import extract_items_from_query
        items = extract_items_from_query("compare Python and Go")
        assert len(items) == 2

    def test_three_way_vs(self):
        from octane.osa.comparison_planner import extract_items_from_query
        items = extract_items_from_query("NVDA vs AMD vs INTC")
        assert len(items) == 3

    def test_or_pattern(self):
        from octane.osa.comparison_planner import extract_items_from_query
        items = extract_items_from_query("PostgreSQL or MySQL")
        assert len(items) == 2

    def test_no_pattern_returns_empty(self):
        from octane.osa.comparison_planner import extract_items_from_query
        items = extract_items_from_query("just a regular question")
        assert len(items) == 0


class TestComparisonPlannerParsing:
    """Tests for ComparisonPlanner JSON parse + fallback."""

    def _make_planner(self, bodega_response: str):
        from octane.osa.comparison_planner import ComparisonPlanner
        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(return_value=bodega_response)
        return ComparisonPlanner(bodega=mock_bodega)

    def _valid_json(self) -> str:
        return json.dumps({
            "items": [
                {"id": "nvda", "label": "NVDA", "canonical_query": "Nvidia stock analysis"},
                {"id": "amd", "label": "AMD", "canonical_query": "AMD stock analysis"},
            ],
            "dimensions": [
                {"id": "valuation", "label": "Valuation", "query_template": "{item} valuation metrics", "priority": 1},
                {"id": "growth", "label": "Growth", "query_template": "{item} revenue growth", "priority": 2},
                {"id": "analyst", "label": "Analyst Targets", "query_template": "{item} analyst price target", "priority": 3},
            ],
        })

    def test_parse_items(self):
        planner = self._make_planner(self._valid_json())
        plan = _run(planner.plan("NVDA vs AMD"))
        assert len(plan.items) == 2
        assert plan.items[0].label == "NVDA"

    def test_parse_dimensions(self):
        planner = self._make_planner(self._valid_json())
        plan = _run(planner.plan("NVDA vs AMD"))
        assert len(plan.dimensions) == 3

    def test_query_for_dimension(self):
        planner = self._make_planner(self._valid_json())
        plan = _run(planner.plan("NVDA vs AMD"))
        item = plan.items[0]   # NVDA
        dim = plan.dimensions[0]  # Valuation
        q = item.query_for_dimension(dim)
        assert "NVDA" in q or "Nvidia" in q

    def test_task_matrix_size(self):
        planner = self._make_planner(self._valid_json())
        plan = _run(planner.plan("NVDA vs AMD"))
        matrix = plan.task_matrix
        # 2 items × 3 dimensions = 6 cells
        assert len(matrix) == 6

    def test_task_matrix_all_pairs(self):
        planner = self._make_planner(self._valid_json())
        plan = _run(planner.plan("NVDA vs AMD"))
        pairs = {(cell[0].id, cell[1].id) for cell in plan.task_matrix}
        assert ("nvda", "valuation") in pairs
        assert ("amd", "growth") in pairs

    def test_sorted_dimensions(self):
        planner = self._make_planner(self._valid_json())
        plan = _run(planner.plan("NVDA vs AMD"))
        priorities = [d.priority for d in plan.sorted_dimensions]
        assert priorities == sorted(priorities)

    def test_fallback_on_empty(self):
        from octane.osa.comparison_planner import ComparisonPlanner
        planner = ComparisonPlanner(bodega=None)
        plan = _run(planner.plan("NVDA vs AMD vs INTC"))
        assert len(plan.items) >= 2
        assert plan.from_llm is False

    def test_to_dict_has_matrix(self):
        planner = self._make_planner(self._valid_json())
        plan = _run(planner.plan("NVDA vs AMD"))
        d = plan.to_dict()
        assert "n_tasks" in d
        assert d["n_tasks"] == 6


# ─────────────────────────────────────────────────────────────────────────────
# ChainParser
# ─────────────────────────────────────────────────────────────────────────────

class TestChainParserBasic:
    """Basic step parsing without any refs."""

    def _parser(self):
        from octane.osa.chain_parser import ChainParser
        return ChainParser()

    def test_single_step_unnamed(self):
        plan = self._parser().parse(["ask What is NVDA?"])
        assert len(plan.steps) == 1
        assert plan.steps[0].name == "step_1"
        assert plan.steps[0].command == "ask"
        assert plan.steps[0].args == "What is NVDA?"

    def test_named_step(self):
        plan = self._parser().parse(["prices: fetch finance NVDA"])
        step = plan.steps[0]
        assert step.name == "prices"
        assert step.command == "fetch"
        assert "NVDA" in step.args

    def test_auto_naming(self):
        plan = self._parser().parse([
            "ask question one",
            "analyze something",
            "synthesize all",
        ])
        names = [s.name for s in plan.steps]
        assert names == ["step_1", "step_2", "step_3"]

    def test_mixed_named_unnamed(self):
        plan = self._parser().parse([
            "intro: ask intro question",
            "search main topic",
        ])
        assert plan.steps[0].name == "intro"
        assert plan.steps[1].name == "step_2"

    def test_step_names_property(self):
        plan = self._parser().parse([
            "a: ask question",
            "b: search topic",
        ])
        assert plan.step_names == {"a", "b"}

    def test_full_text_property(self):
        plan = self._parser().parse(["ask What is the meaning of life"])
        assert "ask" in plan.steps[0].full_text
        assert "meaning" in plan.steps[0].full_text

    def test_empty_chain_raises(self):
        from octane.osa.chain_parser import ChainValidationError
        with pytest.raises(ChainValidationError):
            self._parser().parse([])

    def test_whitespace_only_step_raises(self):
        from octane.osa.chain_parser import ChainValidationError
        with pytest.raises(ChainValidationError):
            self._parser().parse(["   "])


class TestChainParserRefs:
    """Step reference resolution tests."""

    def _parser(self):
        from octane.osa.chain_parser import ChainParser
        return ChainParser()

    def test_has_prev_detected(self):
        plan = self._parser().parse([
            "ask question",
            "synthesize {prev}",
        ])
        assert plan.steps[1].has_prev is True
        assert plan.steps[0].has_prev is False

    def test_has_all_detected(self):
        plan = self._parser().parse([
            "search topic",
            "analyze angle",
            "synthesize {all}",
        ])
        assert plan.steps[2].has_all is True

    def test_named_ref_detected(self):
        plan = self._parser().parse([
            "prices: fetch NVDA",
            "analyze: analyze {prices}",
        ])
        assert "prices" in plan.steps[1].step_refs

    def test_template_var_detected(self):
        plan = self._parser().parse(
            ["ask {{ticker}} earnings"],
            template_vars={"ticker": "NVDA"},
        )
        assert "ticker" in plan.steps[0].template_vars

    def test_forward_ref_raises(self):
        from octane.osa.chain_parser import ChainValidationError
        with pytest.raises(ChainValidationError):
            self._parser().parse([
                "ask {future_step}",
                "future_step: search topic",
            ])

    def test_multi_refs_in_same_step(self):
        plan = self._parser().parse([
            "a: ask question",
            "b: search topic",
            "c: synthesize {a} and {b}",
        ])
        assert plan.steps[2].step_refs == {"a", "b"}


class TestChainParserInterpolation:
    """Tests for ChainStep.interpolate()."""

    def _step(self, index: int, name: str, args: str, step_refs=None, has_prev=False, has_all=False, template_vars=None):
        from octane.osa.chain_parser import ChainStep
        return ChainStep(
            index=index,
            name=name,
            raw=f"{name}: cmd {args}",
            command="ask",
            args=args,
            step_refs=step_refs or set(),
            template_vars=set(template_vars or []),
            has_prev=has_prev,
            has_all=has_all,
        )

    def test_prev_interpolation(self):
        step = self._step(1, "step_2", "summarize {prev}", has_prev=True)
        result = step.interpolate(
            step_outputs={"step_1": "output from step 1"},
        )
        assert "output from step 1" in result

    def test_named_ref_interpolation(self):
        step = self._step(1, "b", "analyze {prices}", step_refs={"prices"})
        result = step.interpolate(
            step_outputs={"prices": "NVDA=450 AMD=180"},
        )
        assert "NVDA=450" in result

    def test_all_interpolation(self):
        step = self._step(2, "report", "synthesize {all}", has_all=True)
        result = step.interpolate(
            step_outputs={"step_1": "output A", "step_2": "output B"},
        )
        assert "output A" in result
        assert "output B" in result

    def test_template_var_interpolation(self):
        step = self._step(0, "intro", "ask about {{ticker}}", template_vars=["ticker"])
        result = step.interpolate(
            step_outputs={},
            template_vars={"ticker": "NVDA"},
        )
        assert "NVDA" in result
        assert "{{ticker}}" not in result

    def test_missing_ref_kept_as_is(self):
        step = self._step(1, "b", "analyze {prices}", step_refs={"prices"})
        result = step.interpolate(step_outputs={})  # prices not in outputs
        assert "{prices}" in result

    def test_missing_template_var_kept(self):
        step = self._step(0, "a", "ask {{missing}}", template_vars=["missing"])
        result = step.interpolate(step_outputs={}, template_vars={})
        assert "{{missing}}" in result


# ─────────────────────────────────────────────────────────────────────────────
# ChainExecutor
# ─────────────────────────────────────────────────────────────────────────────

class TestChainExecutor:
    """Tests for ChainExecutor step dispatch and flow."""

    def _plan(self, step_strings: list[str], template_vars=None):
        from octane.osa.chain_parser import ChainParser
        return ChainParser().parse(step_strings, template_vars=template_vars)

    def _executor(self, bodega_response="mocked output", web_response="web result"):
        from octane.osa.chain import ChainExecutor
        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(return_value=bodega_response)
        return ChainExecutor(bodega=mock_bodega, web_agent=None, orchestrator=None)

    def test_single_ask_step(self):
        plan = self._plan(["ask What is NVDA revenue?"])
        executor = self._executor("$60B annual revenue")
        result = _run(executor.run(plan))
        assert len(result.results) == 1
        assert result.results[0].success

    def test_outputs_accumulated(self):
        plan = self._plan(["ask question", "synthesize {prev}"])
        executor = self._executor("some answer")
        result = _run(executor.run(plan))
        assert len(result.results) == 2

    def test_final_output_is_last_step(self):
        plan = self._plan(["ask question"])
        executor = self._executor("the final answer")
        result = _run(executor.run(plan))
        assert result.final_output == "the final answer"

    def test_failed_step_not_in_successful_steps(self):
        from octane.osa.chain import ChainExecutor
        mock_bodega = MagicMock()
        mock_bodega.chat_simple = AsyncMock(side_effect=RuntimeError("boom"))
        executor = ChainExecutor(bodega=mock_bodega)
        plan = self._plan(["ask question"])
        result = _run(executor.run(plan))
        assert len(result.successful_steps) == 0
        assert result.results[0].error

    def test_outputs_dict(self):
        plan = self._plan(["prices: fetch finance NVDA"])
        executor = self._executor("NVDA=450")
        result = _run(executor.run(plan))
        assert "prices" in result.outputs

    def test_template_var_passes_through(self):
        plan = self._plan(
            ["ask about {{ticker}} earnings"],
            template_vars={"ticker": "AMD"},
        )
        executor = self._executor("AMD earnings are great")
        executor.template_vars = {"ticker": "AMD"}
        calls = []
        original = executor._bodega.chat_simple

        async def capturing(*args, **kwargs):
            calls.append(kwargs.get("prompt", args[0] if args else ""))
            return await original(*args, **kwargs)

        executor._bodega.chat_simple = capturing
        result = _run(executor.run(plan))
        # AMD should appear after interpolation
        assert any("AMD" in str(c) for c in calls)

    def test_run_stream_events_order(self):
        plan = self._plan(["ask question one", "ask question two"])
        executor = self._executor("answer")
        events = _run(_collect_events(executor.run_stream(plan)))
        types = [e["type"] for e in events]
        # Should see: step_start, step_done, step_start, step_done, done
        assert "step_start" in types
        assert "step_done" in types
        assert types[-1] == "done"

    def test_run_stream_done_has_n_steps(self):
        plan = self._plan(["ask question"])
        executor = self._executor("answer")
        events = _run(_collect_events(executor.run_stream(plan)))
        done = [e for e in events if e["type"] == "done"][0]
        assert done["data"]["n_steps"] == 1

    def test_save_chain_creates_file(self, tmp_path):
        plan = self._plan(["ask NVDA earnings", "synthesize {prev}"])
        executor = self._executor("result")
        save_path = str(tmp_path / "my_chain.json")
        result = _run(executor.run(plan, save_path=save_path))
        assert Path(save_path).exists()
        assert result.saved_to == save_path

    def test_saved_chain_json_structure(self, tmp_path):
        plan = self._plan(["prices: fetch NVDA", "report: synthesize {prices}"])
        executor = self._executor("$450")
        save_path = str(tmp_path / "chain.json")
        _run(executor.run(plan, save_path=save_path))
        data = json.loads(Path(save_path).read_text())
        assert data["type"] == "octane_chain"
        assert len(data["steps"]) == 2
        assert data["steps"][0]["name"] == "prices"


# ─────────────────────────────────────────────────────────────────────────────
# InvestigateOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestInvestigateOrchestrator:
    """Tests for InvestigateOrchestrator.run() and run_stream()."""

    def _make_orchestrator(self, plan_response: str = "", synthesis_response: str = "# Report"):
        """Build an InvestigateOrchestrator with fully mocked dependencies."""
        from octane.osa.investigate import InvestigateOrchestrator
        mock_bodega = MagicMock()
        # First call: dimension plan JSON; subsequent calls: synthesis
        import itertools
        call_count = [0]

        async def chat_simple_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return plan_response or json.dumps({
                    "dimensions": [
                        {"id": "d1", "label": "D1", "queries": ["q1"], "priority": 1, "rationale": "r"},
                        {"id": "d2", "label": "D2", "queries": ["q2"], "priority": 2, "rationale": "r"},
                    ]
                })
            return synthesis_response

        mock_bodega.chat_simple = chat_simple_side_effect
        return InvestigateOrchestrator(bodega=mock_bodega, web_agent=None)

    def test_run_returns_result(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("Is NVDA overvalued?"))
        assert result.query == "Is NVDA overvalued?"

    def test_run_returns_report(self):
        orch = self._make_orchestrator(synthesis_response="## Executive Summary\nGreat stock.")
        result = _run(orch.run("test"))
        assert result.report  # Non-empty report
        assert result.word_count > 0

    def test_findings_equal_dimensions(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("test"))
        assert len(result.findings) == 2  # 2 dimensions

    def test_stream_event_types(self):
        orch = self._make_orchestrator()
        events = _run(_collect_events(orch.run_stream("test")))
        types = {e["type"] for e in events}
        assert "plan" in types
        assert "done" in types

    def test_stream_plan_event_has_dimensions(self):
        orch = self._make_orchestrator()
        events = _run(_collect_events(orch.run_stream("test")))
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events) == 1
        assert len(plan_events[0]["data"]["dimensions"]) == 2

    def test_stream_done_event_has_totals(self):
        orch = self._make_orchestrator()
        events = _run(_collect_events(orch.run_stream("test")))
        done = [e for e in events if e["type"] == "done"][0]
        assert "total_ms" in done["data"]
        assert "dimensions_completed" in done["data"]

    def test_no_web_agent_graceful(self):
        """With no web agent, findings degrade gracefully (no crash)."""
        orch = self._make_orchestrator()
        result = _run(orch.run("test graceful degradation"))
        assert result is not None

    def test_result_to_dict(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("test"))
        d = result.to_dict()
        assert "query" in d
        assert "n_dimensions" in d
        assert "total_ms" in d


# ─────────────────────────────────────────────────────────────────────────────
# CompareOrchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareOrchestrator:
    """Tests for CompareOrchestrator.run() and run_stream()."""

    def _make_orchestrator(self, synthesis_response="# Comparison\n\nOverall NVDA wins."):
        from octane.osa.compare import CompareOrchestrator
        mock_bodega = MagicMock()
        call_count = [0]

        async def chat_simple_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Comparison planner response
                return json.dumps({
                    "items": [
                        {"id": "nvda", "label": "NVDA", "canonical_query": "Nvidia analysis"},
                        {"id": "amd", "label": "AMD", "canonical_query": "AMD analysis"},
                    ],
                    "dimensions": [
                        {"id": "val", "label": "Valuation", "query_template": "{item} valuation", "priority": 1},
                        {"id": "growth", "label": "Growth", "query_template": "{item} growth", "priority": 2},
                    ],
                })
            return synthesis_response

        mock_bodega.chat_simple = chat_simple_side_effect
        return CompareOrchestrator(bodega=mock_bodega, web_agent=None)

    def test_run_returns_result(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("NVDA vs AMD"))
        assert result.query == "NVDA vs AMD"

    def test_result_has_report(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("NVDA vs AMD"))
        assert result.report
        assert result.word_count > 0

    def test_cells_equal_matrix_size(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("NVDA vs AMD"))
        # 2 items × 2 dims = 4 cells
        assert len(result.cells) == 4

    def test_cell_for_lookup(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("NVDA vs AMD"))
        cell = result.cell_for("nvda", "val")
        assert cell is not None
        assert cell.item.id == "nvda"
        assert cell.dimension.id == "val"

    def test_cell_for_missing_returns_none(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("NVDA vs AMD"))
        cell = result.cell_for("intc", "val")  # intc not in plan
        assert cell is None

    def test_stream_event_types(self):
        orch = self._make_orchestrator()
        events = _run(_collect_events(orch.run_stream("NVDA vs AMD")))
        types = {e["type"] for e in events}
        assert "plan" in types
        assert "done" in types

    def test_stream_plan_event_has_items_and_dims(self):
        orch = self._make_orchestrator()
        events = _run(_collect_events(orch.run_stream("NVDA vs AMD")))
        plan_events = [e for e in events if e["type"] == "plan"]
        assert len(plan_events) == 1
        data = plan_events[0]["data"]
        assert len(data["items"]) == 2
        assert len(data["dimensions"]) == 2

    def test_no_web_agent_graceful(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("NVDA vs AMD"))
        assert result is not None

    def test_result_to_dict(self):
        orch = self._make_orchestrator()
        result = _run(orch.run("NVDA vs AMD"))
        d = result.to_dict()
        assert "query" in d
        assert "n_cells" in d
        assert "total_ms" in d


# ─────────────────────────────────────────────────────────────────────────────
# CLI — commands registered
# ─────────────────────────────────────────────────────────────────────────────

class TestCLICommandsRegistered:
    """Verify investigate/compare/chain are registered in the Typer app."""

    def _app_command_names(self):
        from octane.main import app
        return {c.callback.__name__ for c in app.registered_commands}

    def test_investigate_registered(self):
        names = self._app_command_names()
        assert "investigate" in names

    def test_compare_registered(self):
        names = self._app_command_names()
        assert "compare" in names

    def test_chain_registered(self):
        names = self._app_command_names()
        assert "chain" in names

    def test_investigate_help(self):
        from octane.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["investigate", "--help"])
        assert result.exit_code == 0
        assert "dimension" in result.output.lower() or "investigat" in result.output.lower()

    def test_compare_help(self):
        from octane.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "compar" in result.output.lower()

    def test_chain_help(self):
        from octane.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["chain", "--help"])
        assert result.exit_code == 0
        assert "step" in result.output.lower() or "pipeline" in result.output.lower()
