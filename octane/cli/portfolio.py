"""octane portfolio — import, view, and analyze your brokerage portfolio."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from octane.cli._shared import console

portfolio_app = typer.Typer(
    name="portfolio",
    help="Import broker CSVs, view positions, and run portfolio analysis.",
    no_args_is_help=True,
)


# ── import ────────────────────────────────────────────────────────────────────

@portfolio_app.command("import")
def portfolio_import(
    path: str = typer.Argument(..., help="Path to broker CSV export file"),
    broker: Optional[str] = typer.Option(None, "--broker", "-b",
        help="Override broker name (Schwab|Fidelity|Vanguard|IBKR|Robinhood|Webull|ETRADE)"),
    account_id: str = typer.Option("", "--account", "-a", help="Account identifier tag"),
    project_id: Optional[int] = typer.Option(None, "--project", "-p", help="Project ID"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and display without saving"),
):
    """Parse a broker CSV export and store positions in Postgres."""
    asyncio.run(_portfolio_import(path, broker, account_id, project_id, dry_run))


async def _portfolio_import(
    path: str,
    broker: str | None,
    account_id: str,
    project_id: int | None,
    dry_run: bool,
) -> None:
    from octane.portfolio.parsers import parse_csv
    from octane.portfolio.store import PortfolioStore

    try:
        positions = parse_csv(path, broker=broker, account_id=account_id)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]PARSE ERROR: {exc}[/]")
        raise typer.Exit(1)

    if not positions:
        console.print("[yellow]No positions found in CSV.[/]")
        return

    detected_broker = positions[0].broker if positions else broker or "Unknown"

    console.print(Rule(f"PORTFOLIO IMPORT — {Path(path).name}"))
    console.print(
        f"  Broker:     [cyan]{detected_broker}[/]\n"
        f"  Positions:  [yellow]{len(positions)}[/]\n"
        f"  Account:    [dim]{account_id or '(none)'}[/]"
    )

    # Preview table
    tbl = Table(show_lines=False, border_style="dim", box=None)
    tbl.add_column("TICKER",   style="cyan", width=10)
    tbl.add_column("QTY",      justify="right", style="white")
    tbl.add_column("AVG COST", justify="right", style="yellow")
    tbl.add_column("COST BASIS", justify="right", style="dim")

    for pos in positions:
        pos.project_id = project_id
        tbl.add_row(
            pos.ticker,
            f"{pos.quantity:,.4f}",
            f"${pos.avg_cost:,.2f}",
            f"${pos.cost_basis:,.2f}",
        )

    console.print(tbl)

    total_basis = sum(p.cost_basis for p in positions)
    console.print(f"\n  [dim]Total cost basis: [yellow]${total_basis:,.2f}[/][/]")

    if dry_run:
        console.print("\n  [dim]Dry run — not saved.[/]")
        return

    store = PortfolioStore()
    await store.connect()
    try:
        count = await store.upsert_many(positions)
        console.print(f"\n  [green]Saved {count} position(s) to Postgres.[/]")
    except Exception as exc:
        console.print(f"  [red]STORE ERROR: {exc}[/]")
    finally:
        await store.close()


# ── show ──────────────────────────────────────────────────────────────────────

@portfolio_app.command("show")
def portfolio_show(
    project_id: Optional[int] = typer.Option(None, "--project", "-p"),
    broker: Optional[str] = typer.Option(None, "--broker", "-b"),
    prices: bool = typer.Option(False, "--prices", help="Fetch live prices from yfinance"),
):
    """Display portfolio positions.  Use --prices to show current market values."""
    asyncio.run(_portfolio_show(project_id, broker, prices))


async def _portfolio_show(
    project_id: int | None,
    broker: str | None,
    prices: bool,
) -> None:
    from octane.portfolio.store import PortfolioStore

    store = PortfolioStore()
    await store.connect()
    try:
        positions = await store.list_positions(project_id=project_id, broker=broker)
    finally:
        await store.close()

    if not positions:
        console.print("[yellow]No positions found. Run `octane portfolio import` first.[/]")
        return

    # Optionally fetch live prices
    live: dict[str, float] = {}
    if prices:
        live = _fetch_prices([p.ticker for p in positions])

    # Filter out info rows that yfinance may mix in
    broker_label = broker or "ALL BROKERS"
    console.print(Rule(f"PORTFOLIO POSITIONS — {broker_label}"))

    tbl = Table(show_lines=False, border_style="dim")
    tbl.add_column("TICKER",    style="cyan",   width=10)
    tbl.add_column("QTY",       justify="right", style="white",  width=12)
    tbl.add_column("AVG COST",  justify="right", style="yellow", width=12)
    tbl.add_column("COST BASIS",justify="right", style="dim",    width=14)
    tbl.add_column("BROKER",    style="dim",     width=12)
    if prices:
        tbl.add_column("CUR PRICE", justify="right", style="green",  width=12)
        tbl.add_column("MKT VALUE", justify="right", style="bright_green", width=14)
        tbl.add_column("P&L",       justify="right", width=14)

    total_basis   = 0.0
    total_market  = 0.0

    for pos in positions:
        row: list[str] = [
            pos.ticker,
            f"{pos.quantity:,.4f}",
            f"${pos.avg_cost:,.2f}",
            f"${pos.cost_basis:,.2f}",
            pos.broker or "-",
        ]
        total_basis += pos.cost_basis

        if prices:
            cur = live.get(pos.ticker)
            if cur is not None:
                mkt = round(pos.quantity * cur, 2)
                pnl = round(mkt - pos.cost_basis, 2)
                sign = "+" if pnl >= 0 else ""
                pct  = round((pnl / pos.cost_basis * 100), 2) if pos.cost_basis else 0.0
                colour = "green" if pnl >= 0 else "red"
                total_market += mkt
                row += [
                    f"${cur:,.2f}",
                    f"${mkt:,.2f}",
                    f"[{colour}]{sign}${pnl:,.2f} ({sign}{pct:.1f}%)[/]",
                ]
            else:
                row += ["-", "-", "-"]

        tbl.add_row(*row)

    console.print(tbl)

    summary = f"\n  Total cost basis:  [yellow]${total_basis:,.2f}[/]"
    if prices and total_market > 0:
        total_pnl = total_market - total_basis
        sign = "+" if total_pnl >= 0 else ""
        colour = "green" if total_pnl >= 0 else "red"
        summary += (
            f"\n  Total market value: [bright_green]${total_market:,.2f}[/]"
            f"\n  Total P&L:          [{colour}]{sign}${total_pnl:,.2f}[/]"
        )
    console.print(summary)


def _fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices via yfinance.  Returns best-effort dict."""
    try:
        import yfinance as yf
        result: dict[str, float] = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).fast_info
                price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
                if price:
                    result[ticker] = float(price)
            except Exception:
                pass
        return result
    except ImportError:
        console.print("[dim]yfinance not installed — skipping live prices[/]")
        return {}


# ── analyze ───────────────────────────────────────────────────────────────────

@portfolio_app.command("analyze")
def portfolio_analyze(
    project_id: Optional[int] = typer.Option(None, "--project", "-p"),
    broker: Optional[str] = typer.Option(None, "--broker", "-b"),
    deep: bool = typer.Option(False, "--deep", help="Run deep multi-agent analysis"),
):
    """Run LLM-based portfolio analysis using stored positions as context."""
    asyncio.run(_portfolio_analyze(project_id, broker, deep))


async def _portfolio_analyze(
    project_id: int | None,
    broker: str | None,
    deep: bool,
) -> None:
    from octane.portfolio.store import PortfolioStore

    store = PortfolioStore()
    await store.connect()
    try:
        positions = await store.list_positions(project_id=project_id, broker=broker)
    finally:
        await store.close()

    if not positions:
        console.print("[yellow]No positions found. Import a CSV first.[/]")
        return

    # Build a structured context string
    lines = ["Portfolio positions:"]
    total_basis = 0.0
    for pos in positions:
        lines.append(
            f"  {pos.ticker}: {pos.quantity:,.2f} shares @ ${pos.avg_cost:,.2f} avg cost"
            + (f" (broker: {pos.broker})" if pos.broker else "")
        )
        total_basis += pos.cost_basis

    lines.append(f"\nTotal cost basis: ${total_basis:,.2f}")
    context = "\n".join(lines)

    query = (
        f"Analyze this investment portfolio. Identify concentration risks, "
        f"sector exposure gaps, and top opportunities to improve risk-adjusted returns. "
        f"Give concrete, actionable recommendations.\n\n{context}"
    )

    console.print(Rule("PORTFOLIO ANALYSIS"))
    console.print(f"[dim]{len(positions)} positions · ${total_basis:,.2f} cost basis[/]\n")

    # Pipe through the OSA orchestrator (same as `octane ask`)
    from octane.osa.orchestrator import Orchestrator
    from octane.daemon.client import DaemonClient

    client = DaemonClient()
    if await client.ping():
        from octane.models.synapse import SynapseDB
        synapse = SynapseDB()
        await synapse.connect()
        orch = Orchestrator(synapse=synapse, deep=deep)
        result = await orch.run(query)
        console.print(result.answer if hasattr(result, "answer") else str(result))
        await synapse.close()
    else:
        # Fallback: print the context and query so user can copy-paste
        console.print(Panel(context, title="PORTFOLIO CONTEXT", border_style="dim"))
        console.print(f"\n[dim]Daemon not running — start with `octane daemon start`.[/]")
        console.print(f"[dim]Or run: octane ask \"{query[:120]}...\"[/]")


# ── risk ──────────────────────────────────────────────────────────────────────

@portfolio_app.command("risk")
def portfolio_risk(
    project_id: Optional[int] = typer.Option(None, "--project", "-p"),
    broker: Optional[str] = typer.Option(None, "--broker", "-b"),
):
    """Show concentration risk, top holdings, and sector breakdown."""
    asyncio.run(_portfolio_risk(project_id, broker))


async def _portfolio_risk(project_id: int | None, broker: str | None) -> None:
    from octane.portfolio.store import PortfolioStore

    store = PortfolioStore()
    await store.connect()
    try:
        positions = await store.list_positions(project_id=project_id, broker=broker)
    finally:
        await store.close()

    if not positions:
        console.print("[yellow]No positions found.[/]")
        return

    # Fetch sector info from yfinance
    sector_map = _fetch_sectors([p.ticker for p in positions])

    # Attach prices for weighting
    prices = _fetch_prices([p.ticker for p in positions])

    weighted: list[tuple[str, float, float, str]] = []  # ticker, mkt_value, pct, sector
    total = 0.0
    for pos in positions:
        price = prices.get(pos.ticker, pos.avg_cost)
        mkt = pos.quantity * price
        sector = sector_map.get(pos.ticker, pos.sector or "Unknown")
        weighted.append((pos.ticker, mkt, 0.0, sector))
        total += mkt

    if total > 0:
        weighted = [(t, v, round(v / total * 100, 2), s) for t, v, _, s in weighted]
    weighted.sort(key=lambda x: -x[1])

    console.print(Rule("PORTFOLIO RISK ANALYSIS"))

    # Top holdings
    tbl = Table(title="TOP HOLDINGS BY WEIGHT", show_lines=False, border_style="cyan")
    tbl.add_column("TICKER",  style="cyan",   width=10)
    tbl.add_column("MKT VALUE", justify="right", style="yellow", width=14)
    tbl.add_column("WEIGHT",  justify="right", style="white",  width=10)
    tbl.add_column("SECTOR",  style="dim",     width=24)

    for ticker, mkt_val, pct, sector in weighted[:15]:
        colour = "red" if pct > 20 else ("yellow" if pct > 10 else "white")
        tbl.add_row(
            ticker,
            f"${mkt_val:,.2f}",
            f"[{colour}]{pct:.1f}%[/]",
            sector,
        )

    console.print(tbl)

    # Sector breakdown
    sector_totals: dict[str, float] = {}
    for _, mkt_val, _, sector in weighted:
        sector_totals[sector] = sector_totals.get(sector, 0.0) + mkt_val

    console.print(Rule("SECTOR EXPOSURE"))
    stbl = Table(show_lines=False, border_style="magenta")
    stbl.add_column("SECTOR",  style="magenta", width=30)
    stbl.add_column("VALUE",   justify="right", style="yellow", width=14)
    stbl.add_column("WEIGHT",  justify="right", style="white",  width=10)

    for sector, val in sorted(sector_totals.items(), key=lambda x: -x[1]):
        pct = round(val / total * 100, 2) if total > 0 else 0.0
        colour = "red" if pct > 30 else ("yellow" if pct > 20 else "white")
        stbl.add_row(sector, f"${val:,.2f}", f"[{colour}]{pct:.1f}%[/]")

    console.print(stbl)

    # Concentration risk flags
    over_weighted = [(t, pct) for t, _, pct, _ in weighted if pct > 10]
    if over_weighted:
        console.print(Rule("CONCENTRATION FLAGS", style="red"))
        for ticker, pct in over_weighted:
            console.print(f"  [red]HIGH CONCENTRATION:[/] {ticker} = {pct:.1f}% of portfolio")

    console.print(f"\n  [dim]Total estimated market value: [yellow]${total:,.2f}[/][/]")


def _fetch_sectors(tickers: list[str]) -> dict[str, str]:
    """Fetch sector info from yfinance.  Best-effort, never raises."""
    try:
        import yfinance as yf
        result: dict[str, str] = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                result[ticker] = info.get("sector", "Unknown")
            except Exception:
                pass
        return result
    except ImportError:
        return {}


# ── rebalance ─────────────────────────────────────────────────────────────────

@portfolio_app.command("rebalance")
def portfolio_rebalance(
    target: str = typer.Option("equal", "--target", "-t",
        help="Target allocation: 'equal' or comma-separated TICKER:WEIGHT list"),
    project_id: Optional[int] = typer.Option(None, "--project", "-p"),
    investment: float = typer.Option(0.0, "--investment", "-i",
        help="New investment amount to factor into rebalancing"),
):
    """Suggest rebalancing trades to hit a target allocation."""
    asyncio.run(_portfolio_rebalance(target, project_id, investment))


async def _portfolio_rebalance(
    target: str,
    project_id: int | None,
    new_investment: float,
) -> None:
    from octane.portfolio.store import PortfolioStore

    store = PortfolioStore()
    await store.connect()
    try:
        positions = await store.list_positions(project_id=project_id)
    finally:
        await store.close()

    if not positions:
        console.print("[yellow]No positions found.[/]")
        return

    prices = _fetch_prices([p.ticker for p in positions])

    # Compute current market values
    holdings: dict[str, float] = {}
    for pos in positions:
        price = prices.get(pos.ticker, pos.avg_cost)
        holdings[pos.ticker] = pos.quantity * price

    total = sum(holdings.values()) + new_investment
    n = len(holdings)

    # Parse target weights
    if target == "equal":
        targets: dict[str, float] = {t: 1.0 / n for t in holdings}
    else:
        parts = [x.strip() for x in target.split(",")]
        raw: dict[str, float] = {}
        for part in parts:
            if ":" in part:
                t, w = part.split(":", 1)
                raw[t.upper()] = float(w)
        total_w = sum(raw.values())
        targets = {t: w / total_w for t, w in raw.items()} if total_w else {}

    if not targets:
        console.print("[red]Could not parse target allocation.[/]")
        return

    console.print(Rule("REBALANCE SUGGESTIONS"))

    tbl = Table(show_lines=False, border_style="dim")
    tbl.add_column("TICKER",       style="cyan",   width=10)
    tbl.add_column("CURRENT",      justify="right", style="dim",    width=14)
    tbl.add_column("TARGET",       justify="right", style="yellow", width=10)
    tbl.add_column("DELTA VALUE",  justify="right", width=16)
    tbl.add_column("ACTION",       justify="right", width=12)

    cur_total = sum(holdings.values())
    for ticker, tgt_weight in sorted(targets.items()):
        tgt_val  = total * tgt_weight
        cur_val  = holdings.get(ticker, 0.0)
        cur_pct  = round(cur_val / cur_total * 100, 1) if cur_total else 0.0
        delta    = tgt_val - cur_val
        sign     = "+" if delta >= 0 else ""
        action   = "BUY" if delta > 0 else ("SELL" if delta < 0 else "HOLD")
        colour   = "green" if action == "BUY" else ("red" if action == "SELL" else "dim")
        tbl.add_row(
            ticker,
            f"${cur_val:,.2f} ({cur_pct:.1f}%)",
            f"{tgt_weight*100:.1f}%",
            f"[{colour}]{sign}${delta:,.2f}[/]",
            f"[{colour}]{action}[/]",
        )

    console.print(tbl)
    console.print(f"\n  [dim]Portfolio value: [yellow]${cur_total:,.2f}[/]  "
                  f"New investment: [yellow]${new_investment:,.2f}[/]  "
                  f"Target total: [yellow]${total:,.2f}[/][/]")
