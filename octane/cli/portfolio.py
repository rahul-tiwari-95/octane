"""octane portfolio — import, view, and analyze your brokerage portfolio."""

from __future__ import annotations

import asyncio
import datetime
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

crypto_app = typer.Typer(
    name="crypto",
    help="Import and view crypto positions.",
    no_args_is_help=True,
)
portfolio_app.add_typer(crypto_app, name="crypto")


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


# ── dividends ─────────────────────────────────────────────────────────────────

@portfolio_app.command("dividends")
def portfolio_dividends(
    project_id: Optional[int] = typer.Option(None, "--project", "-p"),
    broker: Optional[str] = typer.Option(None, "--broker", "-b"),
    save: bool = typer.Option(False, "--save", help="Save dividend data to Postgres"),
):
    """Show dividend schedule, yield, and estimated annual income."""
    asyncio.run(_portfolio_dividends(project_id, broker, save))


async def _portfolio_dividends(
    project_id: int | None, broker: str | None, save: bool
) -> None:
    from octane.portfolio.store import PortfolioStore, DividendStore
    from octane.portfolio.finance import annual_dividend_income
    from octane.portfolio.models import Dividend

    store = PortfolioStore()
    await store.connect()
    try:
        positions = await store.list_positions(project_id=project_id, broker=broker)
    finally:
        await store.close()

    if not positions:
        console.print("[yellow]No positions found.[/]")
        return

    # Fetch dividend info from yfinance
    div_info = _fetch_dividend_info([p.ticker for p in positions])

    holdings = [{"ticker": p.ticker, "quantity": p.quantity} for p in positions]
    result = annual_dividend_income(holdings, div_info)

    console.print(Rule("DIVIDEND INCOME SCHEDULE"))

    tbl = Table(show_lines=False, border_style="dim")
    tbl.add_column("TICKER",     style="cyan",   width=10)
    tbl.add_column("SHARES",     justify="right", style="white",  width=10)
    tbl.add_column("DIV RATE",   justify="right", style="yellow", width=10)
    tbl.add_column("YIELD",      justify="right", style="green",  width=8)
    tbl.add_column("ANNUAL $",   justify="right", style="bright_green", width=12)
    tbl.add_column("EX-DATE",    style="dim",     width=12)

    for item in result["breakdown"]:
        if item["dividend_rate"] == 0 and item["annual_income"] == 0:
            continue
        tbl.add_row(
            item["ticker"],
            f"{item['shares']:,.2f}",
            f"${item['dividend_rate']:.4f}",
            f"{item['dividend_yield']:.2f}%",
            f"${item['annual_income']:,.2f}",
            str(item.get("ex_date") or "-"),
        )

    console.print(tbl)
    console.print(
        f"\n  [bright_green]Estimated annual dividend income: "
        f"${result['total_annual_income']:,.2f}[/]"
    )

    if save and div_info:
        ds = DividendStore()
        await ds.connect()
        try:
            count = 0
            for ticker, info in div_info.items():
                rate = float(info.get("dividendRate", 0) or 0)
                if rate <= 0:
                    continue
                yld = float(info.get("dividendYield", 0) or 0)
                ex_raw = info.get("exDividendDate")
                ex_date = None
                if ex_raw:
                    try:
                        if isinstance(ex_raw, (int, float)):
                            ex_date = datetime.date.fromtimestamp(ex_raw)
                        else:
                            ex_date = datetime.date.fromisoformat(str(ex_raw))
                    except Exception:
                        pass
                div = Dividend(
                    ticker=ticker,
                    amount=rate / 4,  # Assume quarterly
                    ex_date=ex_date,
                    frequency="quarterly",
                    div_yield=yld,
                )
                await ds.upsert_dividend(div)
                count += 1
            console.print(f"  [green]Saved {count} dividend record(s).[/]")
        finally:
            await ds.close()


def _fetch_dividend_info(tickers: list[str]) -> dict[str, dict]:
    """Fetch dividend info from yfinance. Best-effort."""
    try:
        import yfinance as yf
        result: dict[str, dict] = {}
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                if info.get("dividendRate") or info.get("dividendYield"):
                    result[ticker] = {
                        "dividendRate": info.get("dividendRate", 0),
                        "dividendYield": info.get("dividendYield", 0),
                        "exDividendDate": info.get("exDividendDate"),
                        "payoutRatio": info.get("payoutRatio", 0),
                    }
            except Exception:
                pass
        return result
    except ImportError:
        console.print("[dim]yfinance not installed.[/]")
        return {}


# ── lots (tax lot management) ────────────────────────────────────────────────

@portfolio_app.command("lots")
def portfolio_lots(
    ticker: Optional[str] = typer.Argument(None, help="Filter by ticker"),
):
    """Show open tax lots with cost basis, holding period, and gains."""
    asyncio.run(_portfolio_lots(ticker))


async def _portfolio_lots(ticker: str | None) -> None:
    from octane.portfolio.store import TaxLotStore

    store = TaxLotStore()
    await store.connect()
    try:
        lots = await store.list_lots(ticker)
    finally:
        await store.close()

    if not lots:
        console.print("[yellow]No tax lots found. Use `octane portfolio lots-add` to create lots.[/]")
        return

    prices = _fetch_prices(list({lt.ticker for lt in lots}))

    console.print(Rule("TAX LOTS"))

    tbl = Table(show_lines=False, border_style="dim")
    tbl.add_column("ID",        style="dim",    width=5)
    tbl.add_column("TICKER",    style="cyan",   width=8)
    tbl.add_column("SHARES",    justify="right", style="white", width=10)
    tbl.add_column("COST/SH",   justify="right", style="yellow", width=10)
    tbl.add_column("COST BASIS",justify="right", style="dim", width=12)
    tbl.add_column("PURCHASED", style="dim",     width=12)
    tbl.add_column("TERM",      style="dim",     width=6)
    tbl.add_column("UNRL P&L",  justify="right", width=14)

    total_basis = 0.0
    total_pnl = 0.0

    for lt in lots:
        if lt.remaining_shares <= 0:
            continue
        cur_price = prices.get(lt.ticker, lt.cost_per_share)
        cur_val = lt.remaining_shares * cur_price
        pnl = cur_val - lt.cost_basis
        sign = "+" if pnl >= 0 else ""
        colour = "green" if pnl >= 0 else "red"
        term = "LONG" if lt.is_long_term else "SHORT"
        total_basis += lt.cost_basis
        total_pnl += pnl

        tbl.add_row(
            str(lt.id or "-"),
            lt.ticker,
            f"{lt.remaining_shares:,.4f}",
            f"${lt.cost_per_share:,.2f}",
            f"${lt.cost_basis:,.2f}",
            lt.purchase_date.isoformat(),
            term,
            f"[{colour}]{sign}${pnl:,.2f}[/]",
        )

    console.print(tbl)
    sign = "+" if total_pnl >= 0 else ""
    colour = "green" if total_pnl >= 0 else "red"
    console.print(
        f"\n  Total cost basis: [yellow]${total_basis:,.2f}[/]"
        f"  Unrealised P&L: [{colour}]{sign}${total_pnl:,.2f}[/]"
    )


@portfolio_app.command("lots-add")
def portfolio_lots_add(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    shares: float = typer.Argument(..., help="Number of shares"),
    cost: float = typer.Argument(..., help="Cost per share"),
    date: str = typer.Option("today", "--date", "-d", help="Purchase date (YYYY-MM-DD)"),
    broker: str = typer.Option("", "--broker", "-b"),
    account_id: str = typer.Option("", "--account", "-a"),
):
    """Add a tax lot manually."""
    asyncio.run(_lots_add(ticker, shares, cost, date, broker, account_id))


async def _lots_add(
    ticker: str, shares: float, cost: float, date_str: str,
    broker: str, account_id: str,
) -> None:
    from octane.portfolio.store import TaxLotStore
    from octane.portfolio.models import TaxLot

    pd = datetime.date.today() if date_str == "today" else datetime.date.fromisoformat(date_str)
    lot = TaxLot(
        ticker=ticker, shares=shares, cost_per_share=cost,
        purchase_date=pd, broker=broker, account_id=account_id,
    )

    store = TaxLotStore()
    await store.connect()
    try:
        lot_id = await store.add_lot(lot)
        console.print(
            f"  [green]Added lot #{lot_id}: {lot.ticker} {shares:.4f} shares "
            f"@ ${cost:.2f} on {pd.isoformat()}[/]"
        )
    finally:
        await store.close()


@portfolio_app.command("lots-sell")
def portfolio_lots_sell(
    ticker: str = typer.Argument(..., help="Ticker to sell"),
    shares: float = typer.Argument(..., help="Shares to sell"),
    method: str = typer.Option("FIFO", "--method", "-m", help="Cost basis method: FIFO or LIFO"),
    execute: bool = typer.Option(False, "--execute", help="Actually record the sale"),
):
    """Simulate (or execute) a sale using FIFO/LIFO lot allocation."""
    asyncio.run(_lots_sell(ticker, shares, method, execute))


async def _lots_sell(
    ticker: str, shares: float, method: str, execute: bool,
) -> None:
    from octane.portfolio.store import TaxLotStore

    store = TaxLotStore()
    await store.connect()
    try:
        allocations = await store.sell_shares(ticker, shares, method)
    except Exception as exc:
        console.print(f"[red]ERROR: {exc}[/]")
        await store.close()
        return

    if not allocations:
        console.print(f"[yellow]No open lots found for {ticker.upper()}.[/]")
        await store.close()
        return

    prices = _fetch_prices([ticker.upper()])
    cur_price = prices.get(ticker.upper(), 0.0)

    console.print(Rule(f"LOT ALLOCATION — SELL {ticker.upper()} ({method})"))

    tbl = Table(show_lines=False, border_style="dim")
    tbl.add_column("LOT ID",     style="dim",    width=8)
    tbl.add_column("SHARES",     justify="right", style="white", width=10)
    tbl.add_column("COST/SH",    justify="right", style="yellow", width=10)
    tbl.add_column("TERM",       style="dim",     width=6)
    tbl.add_column("EST. GAIN",  justify="right", width=14)

    total_gain = 0.0
    for alloc in allocations:
        gain = (cur_price - alloc["cost_per_share"]) * alloc["shares_sold"] if cur_price > 0 else 0.0
        total_gain += gain
        sign = "+" if gain >= 0 else ""
        colour = "green" if gain >= 0 else "red"
        tbl.add_row(
            str(alloc["lot_id"]),
            f"{alloc['shares_sold']:,.4f}",
            f"${alloc['cost_per_share']:,.2f}",
            "LONG" if alloc["is_long_term"] else "SHORT",
            f"[{colour}]{sign}${gain:,.2f}[/]" if cur_price > 0 else "-",
        )

    console.print(tbl)
    if cur_price > 0:
        sign = "+" if total_gain >= 0 else ""
        colour = "green" if total_gain >= 0 else "red"
        console.print(f"\n  Current price: [cyan]${cur_price:,.2f}[/]  "
                      f"Estimated total gain: [{colour}]{sign}${total_gain:,.2f}[/]")

    if execute:
        for alloc in allocations:
            await store.record_sale(alloc["lot_id"], alloc["shares_sold"])
        console.print(f"  [green]Sale recorded: {shares:.4f} shares of {ticker.upper()}.[/]")
    else:
        console.print(f"\n  [dim]Simulation only — add --execute to record the sale.[/]")

    await store.close()


# ── harvest (tax-loss harvesting) ─────────────────────────────────────────────

@portfolio_app.command("harvest")
def portfolio_harvest(
    project_id: Optional[int] = typer.Option(None, "--project", "-p"),
    broker: Optional[str] = typer.Option(None, "--broker", "-b"),
    min_loss: float = typer.Option(5.0, "--min-loss", help="Minimum loss % to flag"),
):
    """Suggest tax-loss harvesting opportunities."""
    asyncio.run(_portfolio_harvest(project_id, broker, min_loss))


async def _portfolio_harvest(
    project_id: int | None, broker: str | None, min_loss: float,
) -> None:
    from octane.portfolio.store import PortfolioStore
    from octane.portfolio.finance import find_harvest_candidates

    store = PortfolioStore()
    await store.connect()
    try:
        positions = await store.list_positions(project_id=project_id, broker=broker)
    finally:
        await store.close()

    if not positions:
        console.print("[yellow]No positions found.[/]")
        return

    prices = _fetch_prices([p.ticker for p in positions])
    pos_dicts = [{"ticker": p.ticker, "quantity": p.quantity, "avg_cost": p.avg_cost} for p in positions]

    candidates = find_harvest_candidates(pos_dicts, prices, min_loss_pct=min_loss)

    if not candidates:
        console.print("[green]No tax-loss harvesting opportunities found above threshold.[/]")
        return

    console.print(Rule("TAX-LOSS HARVESTING OPPORTUNITIES"))

    tbl = Table(show_lines=False, border_style="dim")
    tbl.add_column("TICKER",    style="cyan",   width=8)
    tbl.add_column("SHARES",    justify="right", style="white", width=10)
    tbl.add_column("COST BASIS",justify="right", style="yellow", width=14)
    tbl.add_column("CUR VALUE", justify="right", style="dim", width=14)
    tbl.add_column("LOSS",      justify="right", style="red", width=14)
    tbl.add_column("LOSS %",    justify="right", style="red", width=8)
    tbl.add_column("WASH RISK", style="dim",     width=6)

    total_harvestable = 0.0
    for c in candidates:
        tbl.add_row(
            c.ticker,
            f"{c.shares:,.2f}",
            f"${c.cost_basis:,.2f}",
            f"${c.current_value:,.2f}",
            f"${c.unrealised_loss:,.2f}",
            f"{c.loss_pct:.1f}%",
            "[red]YES[/]" if c.wash_sale_risk else "[green]NO[/]",
        )
        total_harvestable += c.unrealised_loss

    console.print(tbl)
    console.print(
        f"\n  [red]Total harvestable losses: ${total_harvestable:,.2f}[/]\n"
        f"  [dim]Note: Consult a tax advisor. Wash sale rule applies within 30 days.[/]"
    )


# ── net-worth ────────────────────────────────────────────────────────────────

@portfolio_app.command("net-worth")
def portfolio_net_worth(
    snapshot: bool = typer.Option(False, "--snapshot", help="Save a snapshot to Postgres"),
    history: bool = typer.Option(False, "--history", help="Show net worth timeline"),
    limit: int = typer.Option(30, "--limit", "-n", help="Number of snapshots to show"),
    cash: float = typer.Option(0.0, "--cash", help="Cash position to include"),
):
    """Show or snapshot your net worth across all positions."""
    asyncio.run(_portfolio_net_worth(snapshot, history, limit, cash))


async def _portfolio_net_worth(
    snapshot: bool, history: bool, limit: int, cash: float,
) -> None:
    from octane.portfolio.store import PortfolioStore, CryptoStore, NetWorthStore
    from octane.portfolio.models import NetWorthSnapshot

    if history:
        nw_store = NetWorthStore()
        await nw_store.connect()
        try:
            snaps = await nw_store.list_snapshots(limit=limit)
        finally:
            await nw_store.close()

        if not snaps:
            console.print("[yellow]No snapshots found. Use --snapshot to create one.[/]")
            return

        console.print(Rule("NET WORTH TIMELINE"))
        tbl = Table(show_lines=False, border_style="dim")
        tbl.add_column("DATE",     style="cyan",  width=12)
        tbl.add_column("TOTAL",    justify="right", style="bright_green", width=14)
        tbl.add_column("EQUITIES", justify="right", style="yellow", width=14)
        tbl.add_column("CRYPTO",   justify="right", style="magenta", width=14)
        tbl.add_column("CASH",     justify="right", style="dim", width=14)
        tbl.add_column("POS",      justify="right", style="dim", width=5)

        for s in snaps:
            tbl.add_row(
                s.snapshot_date.isoformat(),
                f"${s.total_value:,.2f}",
                f"${s.equities_value:,.2f}",
                f"${s.crypto_value:,.2f}",
                f"${s.cash_value:,.2f}",
                str(s.position_count),
            )
        console.print(tbl)
        if len(snaps) >= 2:
            change = snaps[0].total_value - snaps[-1].total_value
            sign = "+" if change >= 0 else ""
            colour = "green" if change >= 0 else "red"
            console.print(f"\n  [{colour}]Change over period: {sign}${change:,.2f}[/]")
        return

    # Compute current net worth
    eq_store = PortfolioStore()
    await eq_store.connect()
    try:
        positions = await eq_store.list_positions()
    finally:
        await eq_store.close()

    eq_prices = _fetch_prices([p.ticker for p in positions]) if positions else {}

    equities_value = sum(
        p.quantity * eq_prices.get(p.ticker, p.avg_cost)
        for p in positions
    )

    # Crypto
    crypto_value = 0.0
    crypto_count = 0
    try:
        cr_store = CryptoStore()
        await cr_store.connect()
        try:
            crypto_positions = await cr_store.list_positions()
            if crypto_positions:
                from octane.portfolio.crypto import fetch_crypto_prices
                coins = [cp.coin for cp in crypto_positions]
                crypto_prices = fetch_crypto_prices(coins)
                crypto_value = sum(
                    cp.quantity * crypto_prices.get(cp.coin, cp.cost_per_coin)
                    for cp in crypto_positions
                )
                crypto_count = len(crypto_positions)
        finally:
            await cr_store.close()
    except Exception:
        pass

    total = equities_value + crypto_value + cash

    console.print(Rule("NET WORTH SUMMARY"))
    console.print(f"  Equities:  [yellow]${equities_value:,.2f}[/]  ({len(positions)} positions)")
    console.print(f"  Crypto:    [magenta]${crypto_value:,.2f}[/]  ({crypto_count} coins)")
    console.print(f"  Cash:      [dim]${cash:,.2f}[/]")
    console.print(f"  [bright_green]TOTAL:       ${total:,.2f}[/]")

    if snapshot:
        snap = NetWorthSnapshot(
            total_value=round(total, 2),
            equities_value=round(equities_value, 2),
            crypto_value=round(crypto_value, 2),
            cash_value=round(cash, 2),
            position_count=len(positions) + crypto_count,
        )
        nw_store = NetWorthStore()
        await nw_store.connect()
        try:
            snap_id = await nw_store.save_snapshot(snap)
            console.print(f"  [green]Snapshot #{snap_id} saved for {snap.snapshot_date.isoformat()}.[/]")
        finally:
            await nw_store.close()


# ── xirr ─────────────────────────────────────────────────────────────────────

@portfolio_app.command("xirr")
def portfolio_xirr(
    ticker: str = typer.Argument(..., help="Ticker to compute XIRR for"),
):
    """Compute XIRR (time-weighted return) for a ticker's tax lots."""
    asyncio.run(_portfolio_xirr(ticker))


async def _portfolio_xirr(ticker: str) -> None:
    from octane.portfolio.store import TaxLotStore
    from octane.portfolio.finance import xirr

    store = TaxLotStore()
    await store.connect()
    try:
        lots = await store.list_lots(ticker)
    finally:
        await store.close()

    if not lots:
        console.print(f"[yellow]No tax lots found for {ticker.upper()}. Add lots first.[/]")
        return

    prices = _fetch_prices([ticker.upper()])
    cur_price = prices.get(ticker.upper())

    if not cur_price:
        console.print(f"[yellow]Could not fetch current price for {ticker.upper()}.[/]")
        return

    # Build cashflows: purchases are negative, current value is positive
    cashflows: list[tuple[datetime.date, float]] = []
    total_shares = 0.0
    for lt in lots:
        cashflows.append((lt.purchase_date, -(lt.shares * lt.cost_per_share)))
        total_shares += lt.remaining_shares

    # Current value as a "sell" at today's price
    cashflows.append((datetime.date.today(), total_shares * cur_price))

    rate = xirr(cashflows)
    if rate is None:
        console.print(f"[red]Could not compute XIRR for {ticker.upper()}.[/]")
        return

    console.print(Rule(f"XIRR — {ticker.upper()}"))
    console.print(f"  Time-weighted return: [bright_green]{rate * 100:.2f}%[/] annualised")
    console.print(f"  Based on {len(lots)} lot(s), current price ${cur_price:,.2f}")


# ── crypto import ────────────────────────────────────────────────────────────

@crypto_app.command("import")
def crypto_import_cmd(
    path: str = typer.Argument(..., help="Path to crypto exchange CSV"),
    exchange: Optional[str] = typer.Option(None, "--exchange", "-e",
        help="Override exchange (Coinbase|Kraken|Binance|Gemini)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse without saving"),
):
    """Parse a crypto exchange CSV export and store positions."""
    asyncio.run(_crypto_import(path, exchange, dry_run))


async def _crypto_import(path: str, exchange: str | None, dry_run: bool) -> None:
    from octane.portfolio.crypto import parse_crypto_csv
    from octane.portfolio.store import CryptoStore

    try:
        positions = parse_crypto_csv(path, exchange=exchange)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]PARSE ERROR: {exc}[/]")
        raise typer.Exit(1)

    if not positions:
        console.print("[yellow]No crypto positions found in CSV.[/]")
        return

    detected = positions[0].exchange if positions else "Unknown"
    console.print(Rule(f"CRYPTO IMPORT — {Path(path).name}"))
    console.print(f"  Exchange:   [cyan]{detected}[/]")
    console.print(f"  Positions:  [yellow]{len(positions)}[/]")

    tbl = Table(show_lines=False, border_style="dim", box=None)
    tbl.add_column("COIN",      style="cyan",   width=10)
    tbl.add_column("QTY",       justify="right", style="white")
    tbl.add_column("COST/COIN", justify="right", style="yellow")
    tbl.add_column("COST BASIS",justify="right", style="dim")

    for pos in positions:
        tbl.add_row(
            pos.coin,
            f"{pos.quantity:,.8f}",
            f"${pos.cost_per_coin:,.4f}",
            f"${pos.cost_basis:,.2f}",
        )

    console.print(tbl)
    total = sum(p.cost_basis for p in positions)
    console.print(f"\n  [dim]Total cost basis: [yellow]${total:,.2f}[/][/]")

    if dry_run:
        console.print("\n  [dim]Dry run — not saved.[/]")
        return

    store = CryptoStore()
    await store.connect()
    try:
        count = await store.upsert_many(positions)
        console.print(f"\n  [green]Saved {count} crypto position(s) to Postgres.[/]")
    except Exception as exc:
        console.print(f"  [red]STORE ERROR: {exc}[/]")
    finally:
        await store.close()


# ── crypto show ──────────────────────────────────────────────────────────────

@crypto_app.command("show")
def crypto_show_cmd(
    exchange: Optional[str] = typer.Option(None, "--exchange", "-e"),
    prices: bool = typer.Option(False, "--prices", help="Fetch live prices from CoinGecko"),
):
    """Display crypto positions with optional live pricing."""
    asyncio.run(_crypto_show(exchange, prices))


async def _crypto_show(exchange: str | None, prices_flag: bool) -> None:
    from octane.portfolio.store import CryptoStore

    store = CryptoStore()
    await store.connect()
    try:
        positions = await store.list_positions(exchange=exchange)
    finally:
        await store.close()

    if not positions:
        console.print("[yellow]No crypto positions found. Run `octane portfolio crypto import` first.[/]")
        return

    live: dict[str, float] = {}
    if prices_flag:
        from octane.portfolio.crypto import fetch_crypto_prices
        live = fetch_crypto_prices([p.coin for p in positions])

    console.print(Rule("CRYPTO POSITIONS"))

    tbl = Table(show_lines=False, border_style="dim")
    tbl.add_column("COIN",      style="cyan",   width=10)
    tbl.add_column("QTY",       justify="right", style="white",  width=14)
    tbl.add_column("COST/COIN", justify="right", style="yellow", width=12)
    tbl.add_column("COST BASIS",justify="right", style="dim",    width=14)
    tbl.add_column("EXCHANGE",  style="dim",     width=12)
    if prices_flag:
        tbl.add_column("CUR PRICE", justify="right", style="green",  width=12)
        tbl.add_column("MKT VALUE", justify="right", style="bright_green", width=14)
        tbl.add_column("P&L",       justify="right", width=14)

    total_basis = 0.0
    total_mkt = 0.0

    for pos in positions:
        row: list[str] = [
            pos.coin,
            f"{pos.quantity:,.8f}",
            f"${pos.cost_per_coin:,.4f}",
            f"${pos.cost_basis:,.2f}",
            pos.exchange or "-",
        ]
        total_basis += pos.cost_basis

        if prices_flag:
            cur = live.get(pos.coin)
            if cur:
                mkt = round(pos.quantity * cur, 2)
                pnl = round(mkt - pos.cost_basis, 2)
                sign = "+" if pnl >= 0 else ""
                colour = "green" if pnl >= 0 else "red"
                total_mkt += mkt
                row += [
                    f"${cur:,.2f}",
                    f"${mkt:,.2f}",
                    f"[{colour}]{sign}${pnl:,.2f}[/]",
                ]
            else:
                row += ["-", "-", "-"]

        tbl.add_row(*row)

    console.print(tbl)
    summary = f"\n  Total cost basis: [yellow]${total_basis:,.2f}[/]"
    if prices_flag and total_mkt > 0:
        total_pnl = total_mkt - total_basis
        sign = "+" if total_pnl >= 0 else ""
        colour = "green" if total_pnl >= 0 else "red"
        summary += f"\n  Total market value: [bright_green]${total_mkt:,.2f}[/]"
        summary += f"\n  Total P&L: [{colour}]{sign}${total_pnl:,.2f}[/]"
    console.print(summary)
