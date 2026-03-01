"""Catalyst F1 — price_chart.

Renders a clean price line chart with volume bars from timeseries data
returned by the web finance agent. No LLM, no sandbox, deterministic.

Input (resolved_data keys):
    time_series : list of dicts with keys: timestamp, open, high, low, close, volume
    ticker      : str (optional, falls back to "Asset")

Output dict (AgentResponse.data):
    chart_path  : absolute path to saved PNG
    ticker      : str
    min_price   : float
    max_price   : float
    data_points : int
    summary     : str (human-readable summary for evaluator)
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.price_chart")


def price_chart(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
) -> dict[str, Any]:
    """Render a price chart from upstream timeseries data.

    Returns a result dict with chart_path, summary, and stats.
    Raises RuntimeError if matplotlib is unavailable or data is empty.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — never calls plt.show()
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as e:
        raise RuntimeError(f"matplotlib required for price_chart catalyst: {e}") from e

    rows: list[dict] = resolved_data.get("time_series", [])
    if not rows:
        raise ValueError("price_chart: no time_series data in upstream results")

    # Detect ticker from upstream data (market_data key or _upstream)
    ticker = _extract_ticker(resolved_data)

    # Parse dates and prices
    dates: list[datetime] = []
    closes: list[float] = []
    volumes: list[float] = []

    for row in rows:
        ts = row.get("timestamp", "")
        close = row.get("close")
        volume = row.get("volume", 0)
        if not ts or close is None:
            continue
        try:
            dt = datetime.strptime(str(ts)[:10], "%Y-%m-%d")
        except ValueError:
            continue
        dates.append(dt)
        closes.append(float(close))
        volumes.append(float(volume) if volume else 0.0)

    if len(dates) < 2:
        raise ValueError(f"price_chart: need at least 2 data points, got {len(dates)}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.patch.set_facecolor("#0f0f0f")
    for ax in (ax_price, ax_vol):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#cccccc")
        ax.spines[:].set_color("#333333")

    # Price line
    ax_price.plot(dates, closes, color="#00d4ff", linewidth=2, zorder=3)
    ax_price.fill_between(dates, closes, min(closes), alpha=0.15, color="#00d4ff")
    ax_price.set_ylabel("Price (USD)", color="#cccccc", fontsize=11)
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.2f}"))
    ax_price.grid(True, alpha=0.2, color="#444444")
    ax_price.set_title(
        f"{ticker} — Last {len(dates)} Trading Days",
        color="#ffffff", fontsize=14, fontweight="bold", pad=12,
    )

    # Annotate last price
    last_close = closes[-1]
    first_close = closes[0]
    pct_change = (last_close - first_close) / first_close * 100
    color_arrow = "#00ff88" if pct_change >= 0 else "#ff4444"
    sign = "+" if pct_change >= 0 else ""
    ax_price.annotate(
        f"  ${last_close:,.2f} ({sign}{pct_change:.1f}%)",
        xy=(dates[-1], last_close),
        color=color_arrow, fontsize=10, fontweight="bold",
    )

    # Volume bars
    bar_colors = [
        "#00ff88" if i == 0 or closes[i] >= closes[i - 1] else "#ff4444"
        for i in range(len(closes))
    ]
    ax_vol.bar(dates, volumes, color=bar_colors, alpha=0.7, width=0.8)
    ax_vol.set_ylabel("Volume", color="#cccccc", fontsize=9)
    ax_vol.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1_000_000:.0f}M" if v >= 1_000_000 else f"{v/1_000:.0f}K")
    )
    ax_vol.grid(True, alpha=0.15, color="#444444")

    # X-axis formatting
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_vol.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax_vol.xaxis.get_majorticklabels(), rotation=45, ha="right", color="#cccccc")

    plt.tight_layout(pad=1.5)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{ticker.lower().replace('/', '_')}_price_chart.png"
    chart_path = os.path.join(output_dir, filename)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info(
        "price_chart_saved",
        ticker=ticker,
        data_points=len(dates),
        chart_path=chart_path,
    )

    summary = (
        f"{ticker} price chart saved ({len(dates)} trading days). "
        f"Latest close: ${last_close:,.2f} ({sign}{pct_change:.1f}% from {dates[0].strftime('%b %d')}). "
        f"Range: ${min(closes):,.2f} – ${max(closes):,.2f}. "
        f"Chart: {chart_path}"
    )

    return {
        "chart_path": chart_path,
        "ticker": ticker,
        "min_price": min(closes),
        "max_price": max(closes),
        "last_close": last_close,
        "pct_change": round(pct_change, 2),
        "data_points": len(dates),
        "summary": summary,
        "artifacts": [chart_path],
    }


def _extract_ticker(resolved_data: dict[str, Any]) -> str:
    """Best-effort ticker extraction from resolved data."""
    # Direct keys
    for key in ("ticker", "symbol"):
        v = resolved_data.get(key)
        if v:
            return str(v).upper()
    # Nested in market_data
    md = resolved_data.get("market_data", {})
    if isinstance(md, dict):
        sym = md.get("symbol") or md.get("ticker")
        if sym:
            return str(sym).upper()
    # Nested in _upstream results
    for dep in resolved_data.get("_upstream", {}).values():
        if isinstance(dep, dict):
            sym = dep.get("symbol") or dep.get("ticker")
            if sym:
                return str(sym).upper()
    return "ASSET"
