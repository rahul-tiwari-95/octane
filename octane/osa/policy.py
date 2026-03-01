"""OSA Policy Engine — deterministic rules.

No LLM. Pure Python logic for:
- Max retries
- HITL (human-in-the-loop) triggers
- Allowed/forbidden actions
- Risk level assignment for every DAG node (Session 16)
"""

from __future__ import annotations

import structlog

from octane.models.dag import TaskDAG
from octane.models.decisions import Decision, DecisionLedger

logger = structlog.get_logger().bind(component="osa.policy")

# ── Risk classification tables ────────────────────────────────────────────────

# (agent, sub_agent) → (risk_level, reversible, confidence)
# Tuples with "" sub_agent are catch-all rules for that agent.
_AGENT_RISK_TABLE: dict[tuple[str, str], tuple[str, bool, float]] = {
    # Web reads — always safe
    ("web", "search"):  ("low",    True,  0.95),
    ("web", "news"):    ("low",    True,  0.95),
    ("web", "finance"): ("low",    True,  0.95),
    ("web", ""):        ("low",    True,  0.90),
    # Code execution — potentially irreversible, high impact
    ("code", "full_pipeline"): ("high",   False, 0.80),
    ("code", "execute"):       ("high",   False, 0.80),
    ("code", "write"):         ("medium", True,  0.85),
    ("code", ""):              ("medium", True,  0.85),
    # Memory reads — safe; writes — medium
    ("memory", "read"):  ("low",    True,  0.95),
    ("memory", "write"): ("medium", True,  0.90),
    ("memory", ""):      ("medium", True,  0.90),
    # SysStat reads — safe; model reload — high
    ("sysstat", "monitor"):       ("low",  True,  0.95),
    ("sysstat", "model_reload"):  ("high", False, 0.75),
    ("sysstat", ""):              ("low",  True,  0.90),
    # P&L — preferences are low risk
    ("pnl", ""):         ("low",    True,  0.95),
}

# Keywords in the instruction that bump risk to 'high' or 'critical'
_CRITICAL_KEYWORDS = frozenset([
    "delete", "drop", "destroy", "format", "wipe", "rm -rf",
    "send email", "transfer funds", "make payment",
])
_HIGH_KEYWORDS = frozenset([
    "remove", "overwrite", "execute", "run", "deploy", "publish",
    "push to", "write to disk", "save file",
])


class PolicyEngine:
    """Deterministic rules engine for OSA.

    All rules are pure Python — no LLM inference.
    """

    MAX_RETRIES: int = 3
    MAX_QUERY_LENGTH: int = 10000
    DESTRUCTIVE_KEYWORDS: list[str] = ["delete", "remove", "drop", "destroy", "format"]

    def check_retries(self, current_retries: int) -> bool:
        """Check if retries are within policy limits."""
        return current_retries < self.MAX_RETRIES

    def requires_confirmation(self, query: str) -> bool:
        """Check if the query requires human confirmation."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.DESTRUCTIVE_KEYWORDS)

    def check_query_length(self, query: str) -> bool:
        """Check if query is within length limits."""
        return len(query) <= self.MAX_QUERY_LENGTH

    def assess_dag(self, dag: TaskDAG) -> DecisionLedger:
        """Assess every node in the DAG and return a populated DecisionLedger.

        Each TaskNode becomes one Decision.  Risk level is determined by:
        1. The (agent, sub_agent) risk table.
        2. Keyword scan of the node's instruction — can escalate risk.

        Args:
            dag: The TaskDAG produced by the Decomposer.

        Returns:
            DecisionLedger with one Decision per node, all status='pending'.
        """
        ledger = DecisionLedger(correlation_id=dag.original_query[:40])

        for node in dag.nodes:
            agent = node.agent
            sub_agent = node.metadata.get("sub_agent", "")
            instruction = node.instruction

            # Look up base risk from table (try specific sub_agent first, then catch-all)
            base = (
                _AGENT_RISK_TABLE.get((agent, sub_agent))
                or _AGENT_RISK_TABLE.get((agent, ""))
                or ("low", True, 0.90)
            )
            risk_level, reversible, confidence = base

            # Keyword escalation
            instr_lower = instruction.lower()
            if any(kw in instr_lower for kw in _CRITICAL_KEYWORDS):
                risk_level = "critical"
                reversible = False
                confidence = min(confidence, 0.70)
            elif any(kw in instr_lower for kw in _HIGH_KEYWORDS):
                if risk_level not in ("high", "critical"):
                    risk_level = "high"
                    reversible = False

            decision = Decision(
                correlation_id=dag.original_query[:40],
                action=f"Route to {agent} agent ({sub_agent or 'default'})",
                reasoning=f"Template '{node.metadata.get('template', agent)}' selected by decomposer.",
                risk_level=risk_level,
                confidence=confidence,
                reversible=reversible,
                task_id=node.task_id,
                agent=agent,
                sources=[],
            )
            ledger.add(decision)

            logger.debug(
                "policy_decision",
                task_id=node.task_id,
                agent=agent,
                sub_agent=sub_agent,
                risk=risk_level,
                confidence=confidence,
            )

        return ledger
