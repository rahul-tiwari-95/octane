"""Octane Research — long-running background research workflows.

Architecture:
    ResearchTask  — metadata for a named research topic (Pydantic model)
    ResearchFinding — a single synthesised output from one research cycle
    ResearchStore — Redis metadata + log ring-buffer + Postgres findings storage

Shadows perpetual task:
    octane.tasks.research.research_cycle  — scheduled every N hours via Shadow,
    runs the full OSA pipeline for a topic, stores findings, logs progress.

CLI surface (wired in octane.main):
    octane research start "<topic>" [--every N]
    octane research status
    octane research log  <id>  [--follow]
    octane research report <id>
    octane research stop <id>
"""

from .models import ResearchFinding, ResearchTask
from .store import ResearchStore

__all__ = ["ResearchTask", "ResearchFinding", "ResearchStore"]
