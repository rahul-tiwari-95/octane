"""octane chat, feedback, and session commands."""

from __future__ import annotations

import asyncio

import typer
from rich.panel import Panel
from rich.table import Table

from octane.cli._shared import console, _get_synapse

_CHAT_HELP = """[bold]Slash commands:[/]
  [cyan]/help[/]          — show this message
  [cyan]/trace [id][/]    — show Synapse trace for last response (or a specific id)
  [cyan]/history[/]       — print current conversation history
  [cyan]/clear[/]         — clear conversation history and start fresh
  [cyan]/exit[/]          — end the session (also: exit, quit, q)
"""


def register(app: typer.Typer) -> None:
    app.command()(chat)
    app.command()(feedback)
    app.command()(session)


def chat():
    """💬 Interactive multi-turn chat session with Octane."""
    asyncio.run(_chat())


async def _chat():
    from octane.osa.orchestrator import Orchestrator

    synapse = _get_synapse()
    osa = Orchestrator(synapse, hil_interactive=True)
    session_id = f"chat_{int(__import__('time').time())}"

    conversation_history: list[dict[str, str]] = []
    last_correlation_id: str | None = None

    console.print(Panel(
        "[bold green]Octane Chat[/]\n"
        "[dim]Type your message and press Enter. "
        "Use [bold cyan]/help[/bold cyan] for slash commands, "
        "[bold]exit[/bold] or [bold]quit[/bold] to end the session.[/]",
        border_style="green",
    ))

    with console.status("[dim]Starting up...[/]", spinner="dots"):
        status = await osa.pre_flight(wait_for_bodega=True)

    if status["bodega_reachable"] and status["model_loaded"]:
        model_display = (status.get("model") or "").split("/")[-1] or "model loaded"
        console.print(f"[dim]🧠 {model_display} ready[/]\n")
    else:
        note = status.get("note", "Bodega offline")
        console.print(f"[yellow]⚠ {note}[/]\n")

    turn = 0
    while True:
        try:
            query = console.input("[bold cyan]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/]")
            break

        if not query:
            continue

        if query.startswith("/"):
            parts = query.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "/help":
                console.print(_CHAT_HELP)
                continue

            elif cmd == "/clear":
                conversation_history.clear()
                console.print("[dim]✓ Conversation history cleared.[/]\n")
                continue

            elif cmd == "/history":
                if not conversation_history:
                    console.print("[dim]No history yet.[/]\n")
                else:
                    for i, msg in enumerate(conversation_history, 1):
                        role_tag = "[bold cyan]You:[/]" if msg["role"] == "user" else "[bold green]Octane:[/]"
                        preview = msg["content"][:120].replace("\n", " ")
                        console.print(f"  {i}. {role_tag} {preview}")
                    console.print()
                continue

            elif cmd == "/trace":
                cid = parts[1].strip() if len(parts) > 1 else last_correlation_id
                if cid:
                    _print_synapse_trace(synapse, cid)
                else:
                    console.print("[yellow]No trace available yet — ask something first.[/]\n")
                continue

            else:
                console.print(f"[yellow]Unknown command '{cmd}'. Type /help for options.[/]\n")
                continue

        if query.lower() in ("exit", "quit", "bye", "q"):
            console.print("[dim]Goodbye.[/]")
            break

        turn += 1
        conversation_history.append({"role": "user", "content": query})

        _status = console.status("[dim]⚙  Working...[/]", spinner="dots")
        _status.start()
        response_parts: list[str] = []
        _first = True

        async for chunk in osa.run_stream(
            query,
            session_id=session_id,
            conversation_history=conversation_history,
        ):
            if _first:
                _status.stop()
                console.print(f"\n[bold green]Octane:[/] ", end="")
                _first = False
            console.print(chunk, end="")
            response_parts.append(chunk)

        if _first:
            _status.stop()
        console.print()

        assistant_reply = "".join(response_parts).strip()
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        egress_events = [
            e for e in synapse._events
            if e.event_type == "egress"
        ]
        if egress_events:
            last_correlation_id = egress_events[-1].correlation_id
            console.print(
                f"[dim]  ↳ trace: {last_correlation_id[:16]}…  "
                f"(/trace to inspect)[/]\n"
            )
        else:
            console.print()

        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

    console.print(f"[dim]Session {session_id} — {turn} turn(s)[/]")


def _print_synapse_trace(synapse, correlation_id: str) -> None:
    """Print a compact inline Synapse trace for use inside /trace slash command."""
    events = [e for e in synapse._events if e.correlation_id == correlation_id]
    if not events:
        console.print(f"[yellow]No events found for {correlation_id}[/]\n")
        return
    table = Table(title=f"Trace: {correlation_id[:24]}…", show_header=True, header_style="bold")
    table.add_column("Event", style="cyan", no_wrap=True, width=26)
    table.add_column("Source → Target", width=28)
    table.add_column("Details", overflow="fold")
    for ev in events:
        target = getattr(ev, "target", "—") or "—"
        details = ""
        payload = getattr(ev, "payload", None) or {}
        if isinstance(payload, dict):
            if "agents_used" in payload:
                details = f"agents: {', '.join(payload['agents_used'])}"
            elif "template" in payload:
                details = f"template: {payload['template']}"
            elif "output_preview" in payload:
                details = payload["output_preview"][:60]
        table.add_row(ev.event_type, f"{ev.source} → {target}", details)
    console.print(table)
    console.print()


def feedback(
    signal: str = typer.Argument(..., help="thumbs_up or thumbs_down"),
    trace_id: str = typer.Argument(None, help="Correlation ID from a previous response"),
    user_id: str = typer.Option("default", "--user", "-u", help="User ID"),
):
    """👍 Record feedback on a response to improve future answers.

    Examples:
        octane feedback thumbs_up
        octane feedback thumbs_down <trace_id>
    """
    asyncio.run(_feedback(signal, trace_id, user_id))


async def _feedback(signal: str, trace_id: str | None, user_id: str):
    from octane.osa.router import Router

    valid_signals = {"thumbs_up", "thumbs_down"}
    if signal not in valid_signals:
        console.print(f"[red]Unknown signal '{signal}'. Use: thumbs_up or thumbs_down[/]")
        raise typer.Exit(1)

    synapse = _get_synapse()
    router = Router(synapse)
    pnl_agent = router.get_agent("pnl")

    if not pnl_agent:
        console.print("[red]PnL agent not available.[/]")
        raise typer.Exit(1)

    from octane.models.schemas import AgentRequest
    query = f"feedback {signal}"
    metadata: dict = {"user_id": user_id}
    if trace_id:
        metadata["correlation_id"] = trace_id

    response = await pnl_agent.execute(
        AgentRequest(query=query, source="cli", metadata=metadata)
    )

    score_display = response.data.get("score", "?") if response.data else "?"
    emoji = "👍" if signal == "thumbs_up" else "👎"
    console.print(f"{emoji} [green]{response.output}[/]")

    if isinstance(score_display, int) and score_display == 0:
        console.print("[dim]Preference nudge applied — verbosity updated.[/]")


def session():
    """🧬 Chat session with full annotated replay when you type END.

    Ask questions back and forth. Type END to stop and see the complete
    trace of every agent decision, Redis write, and event for the session.
    """
    asyncio.run(_session())


async def _session():
    from octane.osa.orchestrator import Orchestrator
    from octane.models.synapse import SynapseEventBus

    import time

    synapse = SynapseEventBus()
    osa = Orchestrator(synapse)
    session_id = f"session_{int(time.time())}"

    console.print(Panel(
        "[bold green]Octane Session[/]\n"
        "[dim]Ask anything. Type [bold]END[/bold] when done to see the full replay.[/]",
        border_style="green",
    ))

    with console.status("[dim]Starting up...[/]", spinner="dots"):
        status = await osa.pre_flight(wait_for_bodega=True)

    model_display = (status.get("model") or "").split("/")[-1] or "model loaded"
    if status["bodega_reachable"] and status["model_loaded"]:
        console.print(f"[dim]🧠 {model_display} · session [bold]{session_id}[/bold][/]\n")
    else:
        console.print(f"[yellow]⚠ {status.get('note', 'Bodega offline')}[/]\n")

    turns: list[dict] = []

    turn = 0
    while True:
        try:
            query = console.input("[bold cyan]You:[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session interrupted.[/]")
            break

        if not query:
            continue

        if query.strip().upper() == "END":
            break

        turn += 1
        cids_before = set(synapse.get_all_correlation_ids())

        console.print(f"\n[bold green]Octane:[/] ", end="")
        chunks = []
        async for chunk in osa.run_stream(query, session_id=session_id):
            console.print(chunk, end="")
            chunks.append(chunk)
        console.print("\n")

        cids_after = set(synapse.get_all_correlation_ids())
        new_cids = cids_after - cids_before - {"preflight"}
        cid = next(iter(new_cids)) if new_cids else "unknown"

        turns.append({
            "turn": turn,
            "query": query,
            "correlation_id": cid,
            "output": "".join(chunks).strip(),
        })

    if not turns:
        console.print("[dim]No turns to replay.[/]")
        return

    console.print("\n")
    console.rule("[bold yellow]🧬 SESSION REPLAY[/]", style="yellow")
    console.print(f"[dim]Session ID: {session_id} · {len(turns)} turn(s)[/]\n")

    redis_writes: list[str] = []
    try:
        from octane.tools.redis_client import RedisClient
        redis = RedisClient()
        redis._use_fallback = False
        pattern = f"memory:{session_id}:*"
        redis_keys = await redis.keys_matching(pattern)
        redis_writes = sorted(redis_keys)
    except Exception:
        pass

    for t in turns:
        trace = synapse.get_trace(t["correlation_id"])

        console.print(Panel(
            f"[bold]Turn {t['turn']}[/]  ·  [cyan]{t['query']}[/]\n"
            f"[dim]Trace: {t['correlation_id']}  ·  {trace.total_duration_ms:.0f}ms  ·  "
            f"Agents: {', '.join(a for a in trace.agents_used if a not in ('user', 'osa', ''))}[/]",
            border_style="cyan",
            title=f"[bold cyan]Turn {t['turn']}[/]",
        ))

        event_table = Table(show_lines=True, box=None, padding=(0, 1))
        event_table.add_column("Δt", style="dim", width=9, justify="right")
        event_table.add_column("Event", style="cyan", width=24)
        event_table.add_column("Source → Target", style="green", width=28)
        event_table.add_column("Detail", style="white")

        t0 = trace.started_at
        for event in trace.events:
            if event.correlation_id == "preflight":
                continue

            offset = f"+{(event.timestamp - t0).total_seconds() * 1000:.0f}ms" if t0 else "—"
            src_tgt = f"{event.source}"
            if event.target:
                src_tgt += f" → {event.target}"

            detail = ""
            if event.error:
                detail = f"[red]ERR: {event.error[:80]}[/]"
            elif event.payload:
                for key in ("template", "reasoning", "output_preview", "query",
                            "approach", "agents_used", "tasks_succeeded"):
                    if key in event.payload:
                        val = str(event.payload[key])
                        label = {
                            "template":        "→ template",
                            "reasoning":       "reasoning",
                            "output_preview":  "output",
                            "query":           "query",
                            "approach":        "plan",
                            "agents_used":     "agents",
                            "tasks_succeeded": "succeeded",
                        }.get(key, key)
                        detail = f"[dim]{label}:[/] {val[:90]}"
                        break

            event_table.add_row(offset, event.event_type, src_tgt, detail)

        console.print(event_table)
        console.print(f"\n[bold green]Answer:[/] {t['output']}\n")

    console.rule("[bold magenta]🧠 Redis Memory Written[/]", style="magenta")
    if redis_writes:
        mem_table = Table(show_header=False, box=None, padding=(0, 2))
        mem_table.add_column("Key", style="magenta")
        mem_table.add_column("Value", style="white")

        redis2 = RedisClient()
        for key in redis_writes:
            try:
                raw = await redis2.get(key)
                import json
                val = json.loads(raw) if raw else {}
                answer = val.get("answer", raw or "—")
                if len(answer) > 100:
                    answer = answer[:97] + "..."
                mem_table.add_row(key, answer)
            except Exception:
                mem_table.add_row(key, "[dim](unreadable)[/]")

        console.print(mem_table)
    else:
        console.print("[dim]No memory keys found for this session (Redis may be offline).[/]")

    console.print(f"\n[dim]Session {session_id} complete.[/]")
