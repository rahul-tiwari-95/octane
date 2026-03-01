"""HIL Manager — Human-in-the-Loop decision reviews.

Three trigger categories (from OCTANE_4_PHASE_PLAN_v2.md ADDENDUM):

  BLOCKED     — System literally cannot proceed (captcha, auth, missing data).
                Always asks — no auto-approval possible.

  HIGH-STAKES — Irreversible or consequential action (code execute, external
                write, financial action).
                Asks unless hil_level='relaxed' + decision is medium risk.

  CONFIDENCE  — Accumulated uncertainty exceeds threshold (confidence < 0.60,
                or risk='medium' + confidence < 0.75).
                Asks on 'strict' and 'balanced'; skips on 'relaxed'.

Auto-approval rules:
  relaxed   → approve if risk in ('low', 'medium')
  balanced  → approve if risk='low' AND confidence >= 0.85  (default)
  strict    → approve only if risk='low' AND reversible=True

Session 16: Phase 1 HIL.
  - asyncio.to_thread(input, ...) for non-blocking terminal prompts
  - Rich panels for decision display
  - hil_level read from user_profile (default: 'balanced')
Phase 2: Full Decision Ledger display with structured approve/modify/decline.
Phase 3: P&L auto-approval learning (repeated approvals → auto).
Phase 4: Postgres audit trail of all HIL interactions.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from octane.models.decisions import Decision, DecisionLedger

logger = structlog.get_logger().bind(component="osa.hil")
console = Console()

# Risk level ordering for comparisons
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


class HILManager:
    """Manages Human-in-the-Loop decision reviews.

    Instantiated once per Orchestrator.  ``review_ledger()`` is called
    after PolicyEngine.assess_dag() to resolve all pending decisions.
    """

    def __init__(self, interactive: bool = True) -> None:
        """
        Args:
            interactive: When False, all decisions are auto-approved (useful
                for batch/background/test contexts — no blocking input() calls).
        """
        self._interactive = interactive

    async def review_ledger(
        self,
        ledger: DecisionLedger,
        user_profile: dict[str, Any] | None = None,
    ) -> DecisionLedger:
        """Resolve all pending decisions in the ledger.

        Each decision is either auto-approved (per policy rules) or
        presented to the user for explicit approval/modification/decline.

        Args:
            ledger: The DecisionLedger to review (mutated in place).
            user_profile: Dict optionally containing 'hil_level' key.

        Returns:
            The same ledger with all decisions resolved.
        """
        hil_level = (user_profile or {}).get("hil_level", "balanced")

        for decision in ledger.decisions:
            if decision.is_resolved:
                continue

            if not self._interactive or self._should_auto_approve(decision, hil_level):
                decision.status = "auto_approved"
                logger.debug(
                    "decision_auto_approved",
                    action=decision.action[:80],
                    risk=decision.risk_level,
                    confidence=decision.confidence,
                )
            else:
                # Present to human and collect response
                await self._present_and_collect(decision)

        return ledger

    def _should_auto_approve(self, decision: Decision, hil_level: str) -> bool:
        """Determine if a decision qualifies for automatic approval.

        Levels (from most permissive to strictest):
          relaxed   — auto-approve low + medium risk
          balanced  — auto-approve only low risk with high confidence (default)
          strict    — auto-approve only low risk AND reversible actions
        """
        risk = decision.risk_level
        conf = decision.confidence

        if hil_level == "relaxed":
            return risk in ("low", "medium")
        elif hil_level == "strict":
            return risk == "low" and decision.reversible
        else:  # balanced (default)
            return risk == "low" and conf >= 0.85

    async def _present_and_collect(self, decision: Decision) -> None:
        """Render the decision to the terminal and collect a human response.

        Uses asyncio.to_thread(input, ...) so the event loop is not blocked.
        """
        self._render_decision(decision)

        prompt = (
            "[A]pprove  [M]odify  [D]ecline  → "
        )
        try:
            raw = await asyncio.to_thread(input, prompt)
            choice = raw.strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Non-interactive context (piped stdin, CI) → auto-approve
            decision.status = "auto_approved"
            console.print("[dim]  (no input — auto-approved)[/]")
            return

        if choice.startswith("m"):
            feedback = await asyncio.to_thread(
                input, "  Modification → "
            )
            decision.human_feedback = feedback.strip()
            decision.status = "human_modified"
            console.print(
                f"[green]  ✓ Modified:[/] {decision.human_feedback[:80]}"
            )
        elif choice.startswith("d"):
            reason = await asyncio.to_thread(
                input, "  Reason for declining → "
            )
            decision.human_feedback = reason.strip()
            decision.status = "human_declined"
            console.print(
                f"[red]  ✗ Declined:[/] {decision.human_feedback[:80]}"
            )
        else:
            decision.status = "human_approved"
            console.print("[green]  ✓ Approved[/]")

        logger.info(
            "hil_decision_resolved",
            decision_id=decision.id,
            action=decision.action[:80],
            status=decision.status,
            risk=decision.risk_level,
        )

    def _render_decision(self, decision: Decision) -> None:
        """Render a single Decision as a Rich panel."""
        risk_colours = {
            "low": "green", "medium": "yellow",
            "high": "red", "critical": "bold red",
        }
        colour = risk_colours.get(decision.risk_level, "white")

        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", width=14)
        table.add_column()

        table.add_row("Action:", f"[bold]{decision.action}[/]")
        table.add_row("Risk:", f"[{colour}]{decision.risk_level.upper()}[/]")
        table.add_row(
            "Confidence:",
            f"{decision.confidence:.0%}"
            + (f"  ({decision.uncertainty_reason})" if decision.uncertainty_reason else ""),
        )
        if decision.reasoning:
            table.add_row("Reasoning:", decision.reasoning[:120])
        if decision.code_preview:
            table.add_row("Code:", f"[dim]{decision.code_preview[:200]}[/]")
        if decision.sources:
            table.add_row("Sources:", ", ".join(decision.sources[:3]))
        table.add_row("Reversible:", "Yes" if decision.reversible else "[yellow]No[/]")

        console.print(
            Panel(
                table,
                title=f"[bold]⚡ Decision Review[/]  [{colour}]{decision.risk_level.upper()}[/]",
                border_style=colour,
            )
        )
