"""TaskDAG — the output of OSA's Decomposer.

A user query gets broken into a directed acyclic graph of tasks.
Each task node maps to an agent + sub-task instruction.
Dependencies between nodes define execution order.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class TaskNode(BaseModel):
    """A single task in the decomposed DAG.

    Each node represents one sub-task that will be dispatched to an agent.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent: str = Field(description="Target agent: 'web', 'code', 'memory', 'sysstat', 'pnl'")
    instruction: str = Field(description="What this agent should do")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Task IDs this node depends on (must complete first)",
    )
    priority: int = Field(default=1, description="1=highest priority")
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Hints for the agent: query_type, expected_format, etc.",
    )


class TaskDAG(BaseModel):
    """Directed acyclic graph of tasks produced by OSA.Decomposer.

    The Decomposer analyzes a user query and produces this structure.
    The Router then maps each TaskNode to the appropriate agent instance.
    """

    nodes: list[TaskNode] = Field(default_factory=list)
    reasoning: str = Field(
        default="",
        description="Decomposer's reasoning for this decomposition",
    )
    original_query: str = Field(default="", description="The original user query")

    def get_root_nodes(self) -> list[TaskNode]:
        """Return nodes with no dependencies (can execute immediately)."""
        return [n for n in self.nodes if not n.depends_on]

    def get_dependents(self, task_id: str) -> list[TaskNode]:
        """Return nodes that depend on the given task_id."""
        return [n for n in self.nodes if task_id in n.depends_on]

    def get_node(self, task_id: str) -> TaskNode | None:
        """Get a node by its task_id."""
        for n in self.nodes:
            if n.task_id == task_id:
                return n
        return None

    def execution_order(self) -> list[list[TaskNode]]:
        """Return nodes grouped by execution wave (parallel groups).

        Wave 0: root nodes (no dependencies)
        Wave 1: nodes whose dependencies are all in wave 0
        Wave N: nodes whose dependencies are all in waves < N
        """
        completed: set[str] = set()
        waves: list[list[TaskNode]] = []
        remaining = list(self.nodes)

        while remaining:
            wave = [n for n in remaining if all(d in completed for d in n.depends_on)]
            if not wave:
                # Circular dependency or error — break to avoid infinite loop
                break
            waves.append(wave)
            completed.update(n.task_id for n in wave)
            remaining = [n for n in remaining if n.task_id not in completed]

        return waves
