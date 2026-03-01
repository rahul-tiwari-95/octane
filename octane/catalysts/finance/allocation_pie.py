"""Catalyst F5 — allocation_pie.

Renders a styled donut chart from a portfolio allocation dict
(ticker → weight or dollar value). Current prices from upstream
finance data are used to annotate current values if available.

Input (resolved_data keys):
    price / regularMarketPrice  : current price (float)

Parameters parsed from instruction:
    Ticker allocations are extracted from the query text.

Output dict:
    chart_path  : str
    allocations : dict[str, float]  (ticker → percentage)
    summary     : str
    artifacts   : list[str]
"""

from __future__ import annotations

import os
import re
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.allocation_pie")

# Colours for up to 10 slices
_SLICE_COLORS = [
    "#00d4ff", "#00ff88", "#ff9900", "#ff4488",
    "#cc88ff", "#ffdd00", "#44aaff", "#ff6644",
    "#88ffcc", "#dd44ff",
]

# Pattern: "40% VTI" or "VTI 40%" or "VTI: 40"
_ALLOC_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%?\s*([A-Z]{1,5})|([A-Z]{1,5})\s*[:\s]\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)


def allocation_pie(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
    instruction: str = "",
) -> dict[str, Any]:
    """Render a portfolio allocation donut chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(f"matplotlib required for allocation_pie catalyst: {e}") from e

    allocations = _parse_allocations(instruction)
    if not allocations:
        raise ValueError("allocation_pie: no allocations found in instruction text")

    tickers = list(allocations.keys())
    weights = list(allocations.values())
    total = sum(weights)
    # Normalise to 100%
    pct = [w / total * 100 for w in weights]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    colors = _SLICE_COLORS[: len(tickers)]
    wedges, texts, autotexts = ax.pie(
        pct,
        labels=tickers,
        autopct="%1.1f%%",
        colors=colors,
        pctdistance=0.78,
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "#0f0f0f", "linewidth": 2},
    )

    for text in texts:
        text.set_color("#cccccc")
        text.set_fontsize(11)
        text.set_fontweight("bold")
    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontsize(9)

    ax.set_title(
        "Portfolio Allocation",
        color="#ffffff", fontsize=14, fontweight="bold", pad=16,
    )

    # Legend with percentages
    legend_labels = [f"{t}  {p:.1f}%" for t, p in zip(tickers, pct)]
    ax.legend(
        wedges, legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.12),
        ncol=min(len(tickers), 4),
        facecolor="#1a1a1a", labelcolor="#cccccc", fontsize=9,
        framealpha=0.8,
    )

    plt.tight_layout(pad=1.5)

    os.makedirs(output_dir, exist_ok=True)
    filename = "portfolio_allocation.png"
    chart_path = os.path.join(output_dir, filename)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("allocation_pie_saved", tickers=tickers, chart_path=chart_path)

    alloc_lines = "  " + "\n  ".join(f"{t}: {p:.1f}%" for t, p in zip(tickers, pct))
    summary = f"Portfolio allocation chart:\n{alloc_lines}\n  Chart: {chart_path}"

    return {
        "chart_path": chart_path,
        "allocations": {t: round(p, 2) for t, p in zip(tickers, pct)},
        "summary": summary,
        "artifacts": [chart_path],
    }


def _parse_allocations(text: str) -> dict[str, float]:
    """Extract ticker→weight pairs from instruction text."""
    allocations: dict[str, float] = {}
    for m in _ALLOC_RE.finditer(text):
        if m.group(1) and m.group(2):
            pct = float(m.group(1))
            ticker = m.group(2).upper()
            allocations[ticker] = pct
        elif m.group(3) and m.group(4):
            ticker = m.group(3).upper()
            pct = float(m.group(4))
            allocations[ticker] = pct
    return allocations
