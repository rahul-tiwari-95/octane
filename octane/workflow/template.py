"""WorkflowTemplate — the serializable pipeline format.

A workflow template stores a TaskDAG as plain JSON with optional
``{{variable}}`` placeholders in instruction and metadata fields.
Templates are saved to ``~/.octane/workflows/`` and can be shared,
version-controlled, and replayed with ``octane workflow run``.

Format example
--------------
.. code-block:: json

    {
      "name": "stock-monitor",
      "description": "Fetch and summarise a stock price",
      "variables": {"ticker": "AAPL"},
      "reasoning": "single finance lookup then summarise",
      "nodes": [
        {
          "task_id": "t1",
          "agent": "web",
          "instruction": "Get {{ticker}} current stock price and recent performance",
          "depends_on": [],
          "priority": 1,
          "metadata": {"query_type": "finance"}
        }
      ]
    }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from octane.models.dag import TaskDAG, TaskNode

# Default save directory
WORKFLOW_DIR = Path.home() / ".octane" / "workflows"

# Regex matching {{variable_name}} placeholders
_VAR_RE = re.compile(r"\{\{(\w+)\}\}")


class WorkflowTemplate(BaseModel):
    """A saved, parameterised pipeline template.

    Attributes:
        name:         Short identifier, used as filename stem.
        description:  Human-readable purpose of this workflow.
        variables:    Default values for ``{{placeholders}}``.
                      Keys match placeholder names; values are the defaults
                      used when ``fill()`` is called without overrides.
        reasoning:    Decomposer's original reasoning for this DAG shape.
        nodes:        Serialised ``TaskNode`` dicts with optional placeholders
                      in ``instruction`` and ``metadata`` string values.
    """

    name: str = Field(description="Short workflow identifier")
    description: str = Field(default="", description="What this workflow does")
    variables: dict[str, str] = Field(
        default_factory=dict,
        description="Default variable values for {{placeholder}} substitution",
    )
    reasoning: str = Field(default="", description="Decomposer reasoning captured at export time")
    nodes: list[dict[str, Any]] = Field(
        description="Serialised TaskNode dicts (may contain {{variable}} placeholders)"
    )

    # ── Persistence ───────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "WorkflowTemplate":
        """Load a template from a ``.workflow.json`` file."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path | None = None) -> Path:
        """Save to *path* (default: ``WORKFLOW_DIR/<name>.workflow.json``).

        Creates parent directories if needed.

        Returns:
            The absolute path where the file was written.
        """
        if path is None:
            WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
            path = WORKFLOW_DIR / f"{self.name}.workflow.json"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    # ── Variable substitution ─────────────────────────────────

    def fill(self, overrides: dict[str, str] | None = None) -> list[TaskNode]:
        """Substitute ``{{variables}}`` and return concrete ``TaskNode`` objects.

        Variable resolution order (highest wins):
        1. ``overrides`` — provided at runtime via ``--var``
        2. ``self.variables`` — defaults baked into the template

        Placeholders that have no matching variable are left as-is
        (``{{unknown}}`` stays unchanged) so the user gets a clear
        error from the agent rather than a silent substitution failure.

        Args:
            overrides: Runtime variable values, e.g. ``{"ticker": "MSFT"}``.

        Returns:
            List of ``TaskNode`` instances ready for dispatch.
        """
        resolved: dict[str, str] = {**self.variables, **(overrides or {})}

        nodes: list[TaskNode] = []
        for raw in self.nodes:
            filled = _fill_dict(raw, resolved)
            nodes.append(TaskNode(**filled))
        return nodes

    def to_dag(self, overrides: dict[str, str] | None = None) -> TaskDAG:
        """Fill variables and return a complete ``TaskDAG``.

        Args:
            overrides: Runtime variable overrides.

        Returns:
            A ``TaskDAG`` ready to be passed to ``Orchestrator.run_from_dag()``.
        """
        nodes = self.fill(overrides)
        return TaskDAG(
            nodes=nodes,
            reasoning=self.reasoning,
            original_query=self.variables.get("query", self.description),
        )

    def list_placeholders(self) -> list[str]:
        """Return all unique placeholder names found in this template's nodes."""
        placeholders: set[str] = set()
        for node in self.nodes:
            _collect_placeholders(node, placeholders)
        return sorted(placeholders)


# ── Private helpers ───────────────────────────────────────────────────────────

def _substitute(text: str, variables: dict[str, str]) -> str:
    """Replace ``{{key}}`` with ``variables[key]`` for all matching keys."""
    def replacer(m: re.Match) -> str:
        key = m.group(1)
        return variables.get(key, m.group(0))  # leave unknown vars as-is
    return _VAR_RE.sub(replacer, text)


def _fill_dict(obj: Any, variables: dict[str, str]) -> Any:
    """Recursively substitute variables in strings, dicts, and lists."""
    if isinstance(obj, str):
        return _substitute(obj, variables)
    if isinstance(obj, dict):
        return {k: _fill_dict(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_dict(item, variables) for item in obj]
    return obj


def _collect_placeholders(obj: Any, out: set[str]) -> None:
    """Recursively collect all ``{{name}}`` placeholders from a structure."""
    if isinstance(obj, str):
        out.update(m.group(1) for m in _VAR_RE.finditer(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_placeholders(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_placeholders(item, out)
