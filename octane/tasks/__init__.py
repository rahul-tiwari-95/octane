"""Octane background task registry.

Tasks registered here are picked up by the Shadows Worker subprocess launched via
``octane watch``.  Only perpetual / background tasks live here — the foreground
query pipeline (OSA) remains fully synchronous and is NOT replaced by Shadows.

Public surface
--------------
``octane_tasks``  — TaskCollection consumed by Worker.run(tasks=[...])
"""

from .monitor import monitor_ticker
from .research import research_cycle

# Task collection path consumed by shadows Worker:
#   Worker.run(tasks=["octane.tasks:octane_tasks"])
octane_tasks = [monitor_ticker, research_cycle]

__all__ = ["monitor_ticker", "research_cycle", "octane_tasks"]
