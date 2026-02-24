"""Octane Workflow — shareable pipeline templates.

A workflow template is a saved TaskDAG with ``{{variable}}`` placeholders.
It captures *how* Octane solved a problem (which agents, in what order, with
what instructions) so the same pipeline can be replayed with different inputs.

Public surface
--------------
``WorkflowTemplate``   — Pydantic model, save/load/fill
``export_from_trace``  — build a template from a persisted Synapse trace
``list_workflows``     — list saved ``.workflow.json`` files
``WORKFLOW_DIR``       — default save directory (``~/.octane/workflows/``)
"""

from .template import WorkflowTemplate, WORKFLOW_DIR
from .exporter import export_from_trace, export_from_dag
from .runner import list_workflows

__all__ = [
    "WorkflowTemplate",
    "WORKFLOW_DIR",
    "export_from_trace",
    "export_from_dag",
    "list_workflows",
]
