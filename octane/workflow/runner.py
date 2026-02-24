"""Workflow runner utilities — list saved templates and build DAGs for replay.

The actual execution is done by ``Orchestrator.run_from_dag()``.
This module provides the helpers that the CLI needs to discover and
prepare templates before handing off to the Orchestrator.
"""

from __future__ import annotations

from pathlib import Path

from .template import WorkflowTemplate, WORKFLOW_DIR


def list_workflows(directory: Path | None = None) -> list[Path]:
    """Return paths to all saved ``.workflow.json`` files, newest first.

    Args:
        directory: Directory to scan.  Defaults to ``WORKFLOW_DIR``
                   (``~/.octane/workflows/``).

    Returns:
        List of ``Path`` objects sorted by modification time (newest first).
        Returns an empty list if the directory does not exist.
    """
    d = directory or WORKFLOW_DIR
    if not d.exists():
        return []
    files = sorted(
        d.glob("*.workflow.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return files


def load_workflow(path: Path) -> WorkflowTemplate:
    """Load a ``WorkflowTemplate`` from a file path.

    Thin wrapper around ``WorkflowTemplate.load()`` for consistent imports.

    Args:
        path: Absolute or relative path to a ``.workflow.json`` file.

    Returns:
        Parsed ``WorkflowTemplate``.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the JSON is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")
    return WorkflowTemplate.load(path)
