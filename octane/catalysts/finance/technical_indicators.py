"""Catalyst F4 — technical_indicators.

Computes RSI-14, SMA-20/50, and MACD from timeseries data.
Outputs a signal table and an indicator chart.

Input (resolved_data keys):
    time_series : list of dicts — timestamp, close, volume

Output dict:
    chart_path      : str
    rsi             : float (latest RSI-14)
    rsi_signal      : str ("overbought" | "oversold" | "neutral")
    sma_20          : float
    sma_50          : float | None
    macd            : float
    macd_signal_line: float
    macd_histogram  : float
    summary         : str
    artifacts       : list[str]
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger().bind(component="catalyst.technical_indicators")


def technical_indicators(
    resolved_data: dict[str, Any],
    output_dir: str,
    correlation_id: str = "",
    instruction: str = "",
) -> dict[str, Any]:
    """Compute RSI-14, SMA-20/50, MACD from timeseries data and render chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as e:
        raise RuntimeError(f"matplotlib required for technical_indicators catalyst: {e}") from e

    rows: list[dict] = resolved_data.get("time_series", [])
    if not rows:
        raise ValueError("technical_indicators: no time_series data in upstream results")

    ticker = _extract_ticker(resolved_data)

    dates: list[datetime] = []
    closes: list[float] = []
    for row in rows:
        ts = row.get("timestamp", "")
        close = row.get("close")
        if not ts or close is None:
            continue
        try:
            dt = datetime.strptime(str(ts)[:10], "%Y-%m-%d")
        except ValueError:
            continue
        dates.append(dt)
        closes.append(float(close))

    if len(closes) < 14:
        raise ValueError(f"technical_indicators: need at least 14 data points, got {len(closes)}")

    # ── Compute indicators ────────────────────────────────────────────────────
    rsi_values = _rsi(closes, period=14)
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    macd_line, signal_line, histogram = _macd(closes)

    latest_rsi = rsi_values[-1]
    rsi_signal = "overbought" if latest_rsi > 70 else "oversold" if latest_rsi < 30 else "neutral"
    latest_macd = macd_line[-1] if macd_line else 0.0
    latest_signal = signal_line[-1] if signal_line else 0.0
    latest_hist = histogram[-1] if histogram else 0.0

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10), facecolor="#0f0f0f")
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.05, height_ratios=[3, 1.2, 1.2])

    ax_price = fig.add_subplot(gs[0])
    ax_rsi = fig.add_subplot(gs[1], sharex=ax_price)
    ax_macd = fig.add_subplot(gs[2], sharex=ax_price)

    for ax in (ax_price, ax_rsi, ax_macd):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.spines[:].set_color("#333333")
        ax.grid(True, alpha=0.2, color="#444444")

    # Price + SMAs
    ax_price.plot(dates, closes, color="#00d4ff", linewidth=1.8, label="Close")
    if sma20:
        offset20 = len(dates) - len(sma20)
        ax_price.plot(dates[offset20:], sma20, color="#ffaa00", linewidth=1.2, linestyle="--", label="SMA-20")
    if sma50:
        offset50 = len(dates) - len(sma50)
        ax_price.plot(dates[offset50:], sma50, color="#ff6688", linewidth=1.2, linestyle="--", label="SMA-50")
    ax_price.set_ylabel("Price (USD)", color="#cccccc", fontsize=10)
    ax_price.set_title(f"{ticker} — Technical Indicators", color="#ffffff", fontsize=13, fontweight="bold", pad=10)
    legend = ax_price.legend(facecolor="#2a2a2a", labelcolor="#cccccc", fontsize=8)

    # RSI
    offset_rsi = len(dates) - len(rsi_values)
    ax_rsi.plot(dates[offset_rsi:], rsi_values, color="#cc88ff", linewidth=1.5)
    ax_rsi.axhline(70, color="#ff4444", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_rsi.axhline(30, color="#00ff88", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_rsi.fill_between(dates[offset_rsi:], rsi_values, 70,
                        where=[r > 70 for r in rsi_values], alpha=0.2, color="#ff4444")
    ax_rsi.fill_between(dates[offset_rsi:], rsi_values, 30,
                        where=[r < 30 for r in rsi_values], alpha=0.2, color="#00ff88")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_ylabel("RSI-14", color="#cccccc", fontsize=9)
    ax_rsi.annotate(f"  {latest_rsi:.1f} ({rsi_signal})", xy=(dates[-1], latest_rsi),
                    color="#cc88ff", fontsize=8)

    # MACD
    if macd_line and signal_line:
        offset_macd = len(dates) - len(macd_line)
        offset_signal = len(dates) - len(signal_line)
        offset_hist = len(dates) - len(histogram)
        ax_macd.plot(dates[offset_macd:], macd_line, color="#00d4ff", linewidth=1.2, label="MACD")
        ax_macd.plot(dates[offset_signal:], signal_line, color="#ff9900", linewidth=1.2, label="Signal")
        bar_colors = ["#00ff88" if h >= 0 else "#ff4444" for h in histogram]
        ax_macd.bar(dates[offset_hist:], histogram, color=bar_colors, alpha=0.6, width=0.8)
        ax_macd.axhline(0, color="#555555", linewidth=0.8)
        ax_macd.set_ylabel("MACD", color="#cccccc", fontsize=9)
        ax_macd.legend(facecolor="#2a2a2a", labelcolor="#cccccc", fontsize=7)

    plt.setp(ax_macd.xaxis.get_majorticklabels(), rotation=45, ha="right", color="#cccccc", fontsize=8)
    fig.set_layout_engine("constrained")

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{ticker.lower()}_technical_indicators.png"
    chart_path = os.path.join(output_dir, filename)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info("technical_indicators_saved", ticker=ticker, rsi=latest_rsi, chart_path=chart_path)

    summary = (
        f"{ticker} Technical Analysis:\n"
        f"  RSI-14: {latest_rsi:.1f} → {rsi_signal.upper()}\n"
        f"  SMA-20: ${sma20[-1]:,.2f}" + (f"  SMA-50: ${sma50[-1]:,.2f}" if sma50 else "") + "\n"
        f"  MACD: {latest_macd:.3f}  Signal: {latest_signal:.3f}  Histogram: {latest_hist:.3f}\n"
        f"  Chart: {chart_path}"
    )

    return {
        "chart_path": chart_path,
        "ticker": ticker,
        "rsi": round(latest_rsi, 2),
        "rsi_signal": rsi_signal,
        "sma_20": round(sma20[-1], 2) if sma20 else None,
        "sma_50": round(sma50[-1], 2) if sma50 else None,
        "macd": round(latest_macd, 4),
        "macd_signal_line": round(latest_signal, 4),
        "macd_histogram": round(latest_hist, 4),
        "summary": summary,
        "artifacts": [chart_path],
    }


# ── Indicator implementations ─────────────────────────────────────────────────

def _rsi(closes: list[float], period: int = 14) -> list[float]:
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    results: list[float] = []
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float("inf")
        results.append(100 - 100 / (1 + rs))
    return results


def _sma(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    return [sum(closes[i - period:i]) / period for i in range(period, len(closes) + 1)]


def _ema(closes: list[float], period: int) -> list[float]:
    if len(closes) < period:
        return []
    k = 2 / (period + 1)
    ema = [sum(closes[:period]) / period]
    for price in closes[period:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema


def _macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    if not ema_fast or not ema_slow:
        return [], [], []
    # Align: MACD line starts where both EMAs have values
    offset = slow - fast
    macd_line = [f - s for f, s in zip(ema_fast[offset:], ema_slow)]
    signal_line = _ema(macd_line, signal)
    if not signal_line:
        return macd_line, [], []
    hist_offset = len(macd_line) - len(signal_line)
    histogram = [m - s for m, s in zip(macd_line[hist_offset:], signal_line)]
    return macd_line, signal_line, histogram


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
    return "ASSET"
