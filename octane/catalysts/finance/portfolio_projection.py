"""Catalyst F3 — portfolio_projection.

Monte Carlo simulation: initial investment + monthly DCA contributions
projected over N years using historical return assumptions.
Renders a growth curve chart (10th / 50th / 90th percentile bands).

Input (resolved_data keys):
    price / regularMarketPrice  : current price (from finance agent)

Parameters parsed from instruction:
    initial_investment, monthly_contribution, years, annual_return_pct

Output dict:
    chart_path          : str
    median_final        : float
    p10_final           : float
    p90_final           : float
    years               : int
    monthly_contribution: float
    summary             : str
    artifacts           : list[str]
"""

from __future__ import annotations

import os
import random
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.portfolio_projection")

import re

_NUM_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)|(\b[\d,]+(?:\.\d+)?)")


def portfolio_projection(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
    instruction: str = "",
) -> dict[str, Any]:
    """Monte Carlo portfolio growth projection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise RuntimeError(f"matplotlib/numpy required for portfolio_projection: {e}") from e

    ticker = _extract_ticker(resolved_data)
    initial, monthly, years = _parse_params(instruction)

    # Annual return assumptions: moderate scenario based on historical S&P
    # Mean ~10%, std ~18% (annual), representing broad market ETF behaviour
    annual_mean = 0.10
    annual_std = 0.18
    monthly_mean = annual_mean / 12
    monthly_std = annual_std / (12 ** 0.5)

    n_months = years * 12
    n_sims = 1000

    random.seed(42)
    rng = np.random.default_rng(42)

    # Run simulations
    monthly_returns = rng.normal(monthly_mean, monthly_std, (n_sims, n_months))
    portfolios = np.zeros((n_sims, n_months + 1))
    portfolios[:, 0] = initial

    for m in range(n_months):
        portfolios[:, m + 1] = portfolios[:, m] * (1 + monthly_returns[:, m]) + monthly
    
    # Percentile bands
    p10 = np.percentile(portfolios, 10, axis=0)
    p50 = np.percentile(portfolios, 50, axis=0)
    p90 = np.percentile(portfolios, 90, axis=0)

    time_axis = [i / 12 for i in range(n_months + 1)]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#cccccc")
    ax.spines[:].set_color("#333333")

    ax.fill_between(time_axis, p10, p90, alpha=0.2, color="#00d4ff", label="10th–90th percentile")
    ax.plot(time_axis, p50, color="#00d4ff", linewidth=2.5, label="Median (50th)")
    ax.plot(time_axis, p10, color="#ff9900", linewidth=1, linestyle="--", label="Pessimistic (10th)")
    ax.plot(time_axis, p90, color="#00ff88", linewidth=1, linestyle="--", label="Optimistic (90th)")

    # Annotate endpoints
    for vals, col, label in [(p50, "#00d4ff", "Median"), (p10, "#ff9900", "Low"), (p90, "#00ff88", "High")]:
        ax.annotate(
            f"  {label}: ${vals[-1]:,.0f}",
            xy=(time_axis[-1], vals[-1]),
            color=col, fontsize=9,
        )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlabel("Years", color="#cccccc", fontsize=11)
    ax.set_ylabel("Portfolio Value (USD)", color="#cccccc", fontsize=11)
    ax.set_title(
        f"Portfolio Projection — ${initial:,.0f} initial + ${monthly:,.0f}/mo over {years} years",
        color="#ffffff", fontsize=13, fontweight="bold", pad=12,
    )
    legend = ax.legend(facecolor="#2a2a2a", labelcolor="#cccccc", fontsize=9)
    ax.grid(True, alpha=0.2, color="#444444")

    plt.tight_layout(pad=1.5)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"portfolio_projection_{years}yr.png"
    chart_path = os.path.join(output_dir, filename)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info(
        "portfolio_projection_saved",
        ticker=ticker,
        years=years,
        initial=initial,
        monthly=monthly,
        chart_path=chart_path,
    )

    total_invested = initial + monthly * n_months
    summary = (
        f"Portfolio projection: ${initial:,.0f} initial + ${monthly:,.0f}/month over {years} years.\n"
        f"  Total invested: ${total_invested:,.0f}\n"
        f"  Median final value: ${p50[-1]:,.0f}\n"
        f"  Pessimistic (10th): ${p10[-1]:,.0f}\n"
        f"  Optimistic  (90th): ${p90[-1]:,.0f}\n"
        f"  Chart: {chart_path}"
    )

    return {
        "chart_path": chart_path,
        "ticker": ticker,
        "initial_investment": initial,
        "monthly_contribution": monthly,
        "years": years,
        "total_invested": round(total_invested, 2),
        "p10_final": round(float(p10[-1]), 2),
        "median_final": round(float(p50[-1]), 2),
        "p90_final": round(float(p90[-1]), 2),
        "summary": summary,
        "artifacts": [chart_path],
    }


def _extract_ticker(data: dict[str, Any]) -> str:
    for key in ("ticker", "symbol"):
        v = data.get(key)
        if v:
            return str(v).upper()
    for dep in data.get("_upstream", {}).values():
        if isinstance(dep, dict):
            for key in ("symbol", "ticker"):
                v = dep.get(key)
                if v:
                    return str(v).upper()
    return "PORTFOLIO"


def _parse_params(text: str) -> tuple[float, float, int]:
    """Extract (initial_investment, monthly_contribution, years) from instruction."""
    initial = 500.0
    monthly = 100.0
    years = 20

    if not text:
        return initial, monthly, years

    text_lower = text.lower()

    # Years
    yr_match = re.search(r"(\d+)\s*(?:year|yr)", text_lower)
    if yr_match:
        years = min(int(yr_match.group(1)), 50)

    # Dollar amounts — collect all, pick by context
    dollar_amounts = [float(m[0].replace(",", "")) for m in _NUM_RE.findall(text) if m[0]]

    # Monthly contribution keywords (resolve first to avoid "initial $100/month" ambiguity)
    mo_match = re.search(
        r"\$?\s*([\d,]+(?:\.\d+)?)\s*(?:per\s+month|\/month|monthly|a\s+month|each\s+month)",
        text_lower
    )
    if mo_match:
        monthly = float(mo_match.group(1).replace(",", ""))
    elif len(dollar_amounts) >= 2:
        monthly = dollar_amounts[1]

    # Initial investment keywords — postfix pattern ("$X initial") takes priority to avoid
    # matching "initial $Y/month" when the user wrote "$X initial $Y/month"
    postfix_init = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s+(?:initial(?:ly)?|starting)", text_lower)
    if postfix_init:
        initial = float(postfix_init.group(1).replace(",", ""))
    else:
        init_match = re.search(
            r"(?:start(?:ing)?(?:\s+with)?|initial(?:ly)?|invest(?:ing)?|have|put\s+in)\s+\$?\s*([\d,]+(?:\.\d+)?)",
            text_lower
        )
        if init_match:
            initial = float(init_match.group(1).replace(",", ""))
        elif dollar_amounts:
            # Use the first dollar amount that isn't the monthly contribution
            monthly_val = monthly
            for amt in dollar_amounts:
                if amt != monthly_val:
                    initial = amt
                    break
            else:
                initial = dollar_amounts[0]

    return initial, monthly, years
