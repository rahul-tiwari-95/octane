"""Workflow exporter — build WorkflowTemplate from a Synapse trace or TaskDAG.

Two entry points:

``export_from_trace(correlation_id)``
    Reads the persisted ``.jsonl`` trace file for *correlation_id*, finds the
    ``decomposition_complete`` event, extracts the serialised DAG nodes, and
    produces a ``WorkflowTemplate``.  Requires that the trace was written by
    Octane ≥ Session 12 (the ``dag_nodes_json`` field was added in that session).

``export_from_dag(dag, name)``
    Takes a live ``TaskDAG`` object directly.  Useful for programmatic export
    without needing a trace file (e.g. in tests or when the orchestrator is
    called directly).

Parameterisation
----------------
Both paths attempt light auto-parameterisation: the original query text is
replaced with the ``{{query}}`` placeholder so the template is immediately
reusable with a different query.
"""

from __future__ import annotations

from pathlib import Path

from octane.models.dag import TaskDAG, TaskNode
from octane.models.synapse import SynapseEvent, SynapseEventBus
from .template import WorkflowTemplate, _VAR_RE

# Trace directory (same as SynapseEventBus default)
_TRACE_DIR = Path.home() / ".octane" / "traces"


def export_from_dag(
    dag: TaskDAG,
    name: str,
    description: str = "",
) -> WorkflowTemplate:
    """Build a ``WorkflowTemplate`` from a live ``TaskDAG``.

    The original query is substituted with a ``{{query}}`` placeholder so
    the exported template is immediately parameterisable.

    Args:
        dag: The ``TaskDAG`` to export.
        name: Short identifier for the template (becomes the filename stem).
        description: Optional human-readable description.

    Returns:
        A ``WorkflowTemplate`` (not yet saved to disk — call ``.save()``).
    """
    original_query = dag.original_query or ""

    nodes_raw = []
    for node in dag.nodes:
        raw = node.model_dump()
        # Replace exact original query text with {{query}} placeholder
        if original_query:
            raw["instruction"] = raw["instruction"].replace(original_query, "{{query}}")
            raw["metadata"] = {
                k: v.replace(original_query, "{{query}}") if isinstance(v, str) else v
                for k, v in raw.get("metadata", {}).items()
            }
        nodes_raw.append(raw)

    variables: dict[str, str] = {}
    if original_query:
        variables["query"] = original_query

    return WorkflowTemplate(
        name=name,
        description=description or f"Exported workflow: {name}",
        variables=variables,
        reasoning=dag.reasoning,
        nodes=nodes_raw,
    )


def export_from_trace(
    correlation_id: str,
    name: str | None = None,
    description: str = "",
    trace_dir: Path | None = None,
) -> WorkflowTemplate:
    """Build a ``WorkflowTemplate`` from a persisted Synapse trace.

    Reads ``~/.octane/traces/<correlation_id>.jsonl``, finds the
    ``decomposition_complete`` event, and extracts the full DAG node list
    (requires Octane ≥ Session 12 — the ``dag_nodes_json`` field must be
    present).

    Args:
        correlation_id: The trace ID from ``octane ask``'s footer output.
        name: Template name (defaults to first 8 chars of correlation_id).
        description: Optional description baked into the template.
        trace_dir: Override trace directory (used in tests).

    Returns:
        A ``WorkflowTemplate`` (not yet saved — call ``.save()``).

    Raises:
        FileNotFoundError: If the trace file does not exist.
        ValueError: If the trace has no ``decomposition_complete`` event or
                    the event is missing the ``dag_nodes_json`` field
                    (trace predates Session 12).
    """
    tdir = trace_dir or _TRACE_DIR
    trace_file = tdir / f"{correlation_id}.jsonl"

    if not trace_file.exists():
        raise FileNotFoundError(
            f"No trace found for {correlation_id!r}. "
            f"Run 'octane trace' to list available traces."
        )

    # Load all events
    events: list[SynapseEvent] = []
    with trace_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(SynapseEvent.model_validate_json(line))
                except Exception:
                    continue

    # Find the decomposition_complete event
    decomp = next(
        (e for e in events if e.event_type == "decomposition_complete"),
        None,
    )
    if decomp is None:
        raise ValueError(
            f"Trace {correlation_id!r} has no 'decomposition_complete' event. "
            "The trace may be incomplete or predates workflow export support."
        )

    dag_nodes_json = decomp.payload.get("dag_nodes_json")
    if not dag_nodes_json:
        raise ValueError(
            f"Trace {correlation_id!r} is missing 'dag_nodes_json' — "
            "re-run the query with Octane ≥ Session 12 to generate an exportable trace."
        )

    # Reconstruct TaskDAG
    reasoning = decomp.payload.get("reasoning", "")
    original_query = decomp.payload.get("dag_original_query", "")

    nodes = [TaskNode(**raw) for raw in dag_nodes_json]
    dag = TaskDAG(nodes=nodes, reasoning=reasoning, original_query=original_query)

    template_name = name or correlation_id[:8]
    return export_from_dag(dag, name=template_name, description=description)
