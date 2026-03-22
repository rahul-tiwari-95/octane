"""Octane CLI — thin assembler. All command logic lives in octane/cli/."""

from __future__ import annotations

import typer

from octane.utils import setup_logging
from octane.cli import register_all

# ── Backward-compat re-exports (tests import these from octane.main) ──────────
# trace module
from octane.cli.trace import (  # noqa: F401
    _resolve_trace_id,
    _COLOUR as _EVENT_COLOURS,
    _LABEL  as _EVENT_ICONS,
    trace,
    _trace,
)
from octane.cli.trace import _print_verbose_web_trace  # noqa: F401  (removed, kept as no-op)
# ask module
from octane.cli.ask import ask, _ask  # noqa: F401
# pref module
from octane.cli.pref import _PREF_CHOICES  # noqa: F401

setup_logging()

app = typer.Typer(
    name="octane",
    help="🔥 Octane — Local-first agentic OS for Apple Silicon",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

register_all(app)

if __name__ == "__main__":
    app()
