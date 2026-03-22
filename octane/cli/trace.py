"""octane trace command — technical event timeline."""

from __future__ import annotations

import asyncio

import typer
from rich.rule import Rule

from octane.cli._shared import console, _get_synapse

# Short uppercase labels per event type (no emojis)
_LABEL: dict[str, str] = {
    "ingress":                "INGRESS",
    "guard":                  "GUARD",
    "decomposition":          "DECOMPOSE",
    "decomposition_complete": "DECOMPOSE",
    "dispatch":               "DISPATCH",
    "agent_start":            "AGENT_START",
    "agent_complete":         "AGENT_DONE",
    "memory_read":            "MEM_READ",
    "memory_write":           "MEM_WRITE",
    "preflight":              "PREFLIGHT",
    "egress":                 "EGRESS",
    "web_search_round":       "WEB_SEARCH",
    "web_depth_analysis":     "DEPTH_ANAL",
    "msr_decision":           "MSR_DECIDE",
    "msr_answers":            "MSR_STEER",
    "web_synthesis":          "WEB_SYNTH",
}

_COLOUR: dict[str, str] = {
    "ingress":                "bold white",
    "guard":                  "bold yellow",
    "decomposition":          "cyan",
    "decomposition_complete": "cyan",
    "dispatch":               "green",
    "agent_start":            "bold green",
    "agent_complete":         "bold green",
    "memory_read":            "blue",
    "memory_write":           "blue",
    "preflight":              "dim",
    "egress":                 "bold magenta",
    "web_search_round":       "bold cyan",
    "web_depth_analysis":     "bold blue",
    "msr_decision":           "bold yellow",
    "msr_answers":            "yellow",
    "web_synthesis":          "bold magenta",
}

_WATERFALL_WIDTH = 58  # chars for the bar section


def register(app: typer.Typer) -> None:
    app.command()(trace)


def trace(
    correlation_id: str = typer.Argument(
        None,
        help="Correlation ID to inspect. Partial IDs accepted (first 8+ chars). "
             "Omit to list recent traces.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Expand URL lists and full content previews inside each web event.",
    ),
    list_mode: bool = typer.Option(
        False, "--list", "-l",
        help="List recent traces (alias for running with no argument).",
    ),
):
    """Technical event-stream timeline for a query execution lifecycle."""
    asyncio.run(_trace(None if list_mode else correlation_id, verbose=verbose))


# ── Main dispatch ─────────────────────────────────────────────────────────────

async def _trace(correlation_id: str | None, verbose: bool = False):
    synapse = _get_synapse()
    if correlation_id:
        resolved = _resolve_trace_id(synapse, correlation_id)
        if resolved is None:
            console.print(f"[yellow]No trace found matching: [bold]{correlation_id}[/]")
            console.print("[dim]Run 'octane trace' with no args to list recent traces.[/]")
            return
        _print_trace_detail(synapse.get_trace(resolved), verbose=verbose)
    else:
        _print_trace_list(synapse)


# ── Detail view ───────────────────────────────────────────────────────────────

def _print_trace_detail(t, verbose: bool) -> None:
    real_events = [e for e in t.events if e.event_type != "preflight"]
    if not real_events:
        console.print("[yellow]Trace exists but has no events.[/]")
        return

    t0 = t.started_at
    total_ms = t.total_duration_ms

    ingress = next((e for e in real_events if e.event_type == "ingress"), None)
    egress  = next((e for e in real_events if e.event_type == "egress"),  None)

    query_text = ""
    if ingress and ingress.payload:
        query_text = (
            ingress.payload.get("query")
            or ingress.payload.get("q")
            or ingress.payload.get("text")
            or ""
        )

    dag_reason = (egress.payload or {}).get("dag_reasoning", "") if egress else ""
    started_str = t0.strftime("%Y-%m-%d %H:%M:%S UTC") if t0 else "?"
    ended_str = egress.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC") if (egress and egress.timestamp) else ""

    visible_agents = [a for a in t.agents_used
                      if a not in ("user", "osa", "osa.decomposer", "osa.evaluator")]
    status_str = "[green]ok[/]" if t.success else "[red]failed[/]"

    # ── Header ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(f"TRACE  {t.correlation_id}", style="bold blue"))
    if query_text:
        console.print(f"  [bold]query[/]    {query_text}")
    console.print(f"  [bold]started[/]  {started_str}")
    if ended_str:
        console.print(f"  [bold]ended[/]    {ended_str}  (+{total_ms:.0f}ms)")
    if dag_reason:
        console.print(f"  [bold]routing[/]  {dag_reason}")
    agents_str = "  ".join(visible_agents) if visible_agents else "(osa only)"
    console.print(f"  [bold]agents[/]   {agents_str}   [bold]status[/] {status_str}")

    # ── Event stream ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule(f"EVENTS  ({len(real_events)} total)", style="dim"))

    for i, event in enumerate(real_events):
        offset_ms = (
            (event.timestamp - t0).total_seconds() * 1000
            if (t0 and event.timestamp) else 0.0
        )
        label  = _LABEL.get(event.event_type, event.event_type.upper())
        colour = _COLOUR.get(event.event_type, "white")

        # source -> target string
        src_tgt = event.source or ""
        if event.target:
            src_tgt += f" -> {event.target}"

        console.print()
        console.print(
            f"  [dim]T+{offset_ms:<9.0f}[/]  [{colour}]{label:<12}[/]  [dim]{src_tgt}[/]"
        )

        if event.error:
            console.print(f"               [red]  ERROR: {event.error}[/]")

        if event.payload:
            _print_payload(event.event_type, event.payload, verbose=verbose)

    # ── Waterfall ────────────────────────────────────────────────────────────
    if total_ms > 0 and t0:
        console.print()
        console.print(Rule(f"WATERFALL  (0ms -> {total_ms:.0f}ms)", style="dim"))
        console.print()
        _print_waterfall(real_events, t0, total_ms)

    console.print()
    console.print(
        f"  [dim]{len(real_events)} events  ·  {total_ms:.0f}ms total[/]  "
        "[dim]  octane trace <id> -v  — expand URL lists[/]"
    )
    console.print()


def _print_payload(event_type: str, payload: dict, *, verbose: bool) -> None:
    """Print payload fields as indented continuation lines under the event header."""
    ind = "               "  # aligns under the label column

    if event_type == "ingress":
        q = payload.get("query") or payload.get("q") or payload.get("text") or ""
        if q:
            console.print(f"{ind}  [dim]query:[/] {q}")

    elif event_type in ("decomposition", "decomposition_complete"):
        route  = payload.get("routing") or payload.get("agent") or payload.get("template") or ""
        reason = payload.get("reasoning") or ""
        if route:
            console.print(f"{ind}  [dim]route:[/]  {route}")
        if reason:
            console.print(f"{ind}  [dim]reason:[/] {reason[:200]}")

    elif event_type == "agent_start":
        agent = payload.get("agent") or payload.get("agents_used") or ""
        if agent:
            console.print(f"{ind}  [dim]agent:[/] {agent}")

    elif event_type == "web_search_round":
        rnd   = payload.get("round", "?")
        n_url = payload.get("urls_found", 0)
        n_ext = payload.get("pages_extracted", 0)
        console.print(
            f"{ind}  [dim]round:[/] {rnd}   [dim]urls:[/] {n_url} found  {n_ext} extracted"
        )

        queries = payload.get("queries", [])
        if queries:
            console.print(f"{ind}  [dim]queries:[/]")
            for q in queries:
                console.print(f"{ind}    [dim]·[/] {q}")

        extracted_detail = payload.get("extracted_detail", [])
        all_urls         = payload.get("urls", [])

        if verbose and all_urls:
            extracted_set = {d.get("url", "") for d in extracted_detail}
            console.print(f"{ind}  [dim]all urls ({len(all_urls)}):[/]")
            for url in all_urls[:50]:
                tag = "[green]extracted[/]" if url in extracted_set else "[dim]skipped  [/]"
                console.print(f"{ind}    {tag}  {url[:100]}")
            if len(all_urls) > 50:
                console.print(f"{ind}    [dim]+{len(all_urls) - 50} more[/]")

        if extracted_detail:
            limit = None if verbose else 5
            console.print(f"{ind}  [dim]extracted pages:[/]")
            for d in (extracted_detail if verbose else extracted_detail[:5]):
                chars  = d.get("chars", 0)
                words  = d.get("words", 0)
                method = d.get("method", "?")
                console.print(
                    f"{ind}    [dim]·[/] {d.get('url','?')[:90]}"
                    f"  [dim][{method}, {chars:,}c, {words}w][/]"
                )
            if not verbose and len(extracted_detail) > 5:
                console.print(
                    f"{ind}    [dim]+{len(extracted_detail) - 5} more  (use -v to expand)[/]"
                )

    elif event_type == "web_depth_analysis":
        rnd      = payload.get("round", "?")
        followups = payload.get("followup_queries", [])
        uctx     = payload.get("user_context") or ""
        console.print(
            f"{ind}  [dim]round:[/] {rnd}   [dim]follow-up queries:[/] {len(followups)}"
        )
        if followups:
            for fq in followups[:10]:
                q_text   = fq.get("query", fq) if isinstance(fq, dict) else fq
                rational = fq.get("rationale", "") if isinstance(fq, dict) else ""
                line = f"{ind}    [dim]·[/] {str(q_text)[:100]}"
                if rational:
                    line += f"  [dim]({rational[:60]})[/]"
                console.print(line)
            if len(followups) > 10:
                console.print(f"{ind}    [dim]+{len(followups) - 10} more[/]")
        if uctx:
            console.print(f"{ind}  [dim]steering:[/] {uctx[:200]}")

    elif event_type == "msr_decision":
        ask  = payload.get("should_ask", False)
        n_q  = payload.get("n_questions", 0)
        console.print(f"{ind}  [dim]ask:[/] {ask}   [dim]questions:[/] {n_q}")
        if ask:
            for q in payload.get("questions", []):
                console.print(f"{ind}    [dim]·[/] {q}")

    elif event_type == "msr_answers":
        ctx = str(payload.get("user_context", ""))
        if ctx:
            console.print(f"{ind}  [dim]steering:[/] {ctx[:300]}")

    elif event_type == "web_synthesis":
        n_art    = payload.get("n_articles", 0)
        mode     = payload.get("mode", "")
        tok      = payload.get("tokens_approx") or payload.get("tokens") or ""
        tok_part = f"   [dim]tokens:[/] ~{tok}" if tok else ""
        console.print(f"{ind}  [dim]articles:[/] {n_art}   [dim]mode:[/] {mode}{tok_part}")
        preview = payload.get("output_preview") or payload.get("output") or ""
        if preview:
            console.print(f"{ind}  [dim]output:[/] {str(preview)[:400]}")

    elif event_type in ("agent_complete", "egress"):
        preview = payload.get("output_preview") or payload.get("output") or ""
        agents  = payload.get("agents_used") or ""
        if agents:
            console.print(f"{ind}  [dim]agents:[/] {agents}")
        if preview:
            console.print(f"{ind}  [dim]output:[/] {str(preview)[:400]}")

    elif event_type in ("memory_read", "memory_write"):
        key = payload.get("key") or payload.get("query") or ""
        hit = payload.get("hit")
        if key:
            console.print(f"{ind}  [dim]key:[/] {key[:80]}")
        if hit is not None:
            console.print(f"{ind}  [dim]hit:[/] {hit}")

    else:
        for k, v in payload.items():
            if v not in (None, "", [], {}):
                console.print(f"{ind}  [dim]{k}:[/] {str(v)[:200]}")


def _print_waterfall(events, t0, total_ms: float) -> None:
    """ASCII bar chart: each event's fire-time as a horizontal bar from T=0."""
    w = _WATERFALL_WIDTH
    ms_per_char = total_ms / w
    label_w = 13

    console.print(f"  {'':>{label_w}}  0{'─' * (w - 2)}{total_ms:.0f}ms")

    for event in events:
        if not event.timestamp:
            continue
        offset_ms = (event.timestamp - t0).total_seconds() * 1000
        pos   = min(int(offset_ms / ms_per_char), w)
        label = _LABEL.get(event.event_type, event.event_type.upper())[:label_w]
        colour = _COLOUR.get(event.event_type, "white")
        bar   = "─" * pos + "+"
        console.print(
            f"  [{colour}]{label:>{label_w}}[/]  [dim]{bar:<{w+1}}[/]  [dim]{offset_ms:.0f}[/]"
        )


# ── List view ─────────────────────────────────────────────────────────────────

def _print_trace_list(synapse) -> None:
    trace_ids = synapse.list_traces(limit=20)
    if not trace_ids:
        console.print("[yellow]No traces found. Run 'octane ask' first.[/]")
        return

    console.print()
    console.print(Rule("Recent Traces  (~/.octane/traces/)", style="bold blue"))

    for cid in trace_ids:
        if cid == "preflight":
            continue
        t = synapse.get_trace(cid)
        real_events = [e for e in t.events if e.event_type != "preflight"]

        ingress = next((e for e in real_events if e.event_type == "ingress"), None)
        query_text = ""
        if ingress and ingress.payload:
            query_text = (
                ingress.payload.get("query")
                or ingress.payload.get("q")
                or ingress.payload.get("text")
                or ""
            )

        started_str = t.started_at.strftime("%m-%d %H:%M:%S UTC") if t.started_at else "?"
        visible_agents = [a for a in t.agents_used
                          if a not in ("user", "osa", "osa.decomposer", "osa.evaluator")]
        status_str = "[green]ok[/]" if t.success else "[red]fail[/]"
        dur_ms = t.total_duration_ms
        dur_str = f"{dur_ms / 1000:.1f}s" if dur_ms >= 1000 else f"{dur_ms:.0f}ms"
        agents_str = ", ".join(visible_agents) or "osa"

        console.print(
            f"\n  [cyan]{t.correlation_id}[/]\n"
            f"  [dim]  {started_str}  ·  {len(real_events)} events  ·  {dur_str}  ·  {agents_str}  ·  [/]{status_str}"
        )
        if query_text:
            console.print(f"  [dim]  q: {query_text[:120]}[/]")

    console.print()
    console.print("  [dim]octane trace <id>       — full event timeline[/]")
    console.print("  [dim]octane trace <id> -v    — with all URLs and content previews[/]")
    console.print("  [dim]partial IDs accepted (first 8+ chars)[/]")
    console.print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_trace_id(synapse, partial_id: str) -> str | None:
    """Return a full correlation ID that starts with partial_id, or None."""
    t = synapse.get_trace(partial_id)
    if t.events:
        return partial_id
    all_ids = synapse.list_traces(limit=50)
    for cid in all_ids:
        if cid.startswith(partial_id):
            return cid
    return None


# ── Backward-compat stubs (imported by octane/main.py for test re-exports) ───

def _print_verbose_web_trace(events, t0) -> None:  # noqa: ARG001
    """Removed — payload detail is now shown inline in the event stream."""
    pass


def _format_agent_tags(agents_used: list[str]) -> str:
    parts = [a for a in agents_used
             if a not in ("user", "osa", "osa.decomposer", "osa.evaluator")]
    return "  ".join(parts) if parts else "(osa only)"

    if correlation_id:
        resolved = _resolve_trace_id(synapse, correlation_id)
        if resolved is None:
            console.print(f"[yellow]No trace found matching: [bold]{correlation_id}[/bold][/]")
            console.print("[dim]Run 'octane trace' (no args) to list recent traces.[/]")
            return

        t = synapse.get_trace(resolved)
        real_events = [e for e in t.events if e.correlation_id != "preflight"]

        if not real_events:
            console.print(f"[yellow]Trace [bold]{resolved}[/bold] exists but has no events.[/]")
            return

        started_str = t.started_at.strftime("%Y-%m-%d %H:%M:%S UTC") if t.started_at else "?"
        agent_tags = _format_agent_tags(t.agents_used)

        egress = next((e for e in real_events if e.event_type == "egress"), None)
        dag_nodes_raw = egress.payload.get("dag_nodes_json", "") if egress and egress.payload else ""
        dag_reason = egress.payload.get("dag_reasoning", "") if egress and egress.payload else ""

        header_lines = [
            f"[bold]Trace ID:[/]  {t.correlation_id}",
            f"[bold]Started:[/]   {started_str}",
            f"[bold]Duration:[/]  {t.total_duration_ms:.0f} ms",
            f"[bold]Status:[/]    {'[green]✅ success[/]' if t.success else '[red]❌ failed[/]'}",
            f"[bold]Agents:[/]    {agent_tags}",
        ]
        if dag_reason:
            header_lines.append(f"[bold]Routing:[/]   [dim]{dag_reason[:100]}[/]")

        console.print(Panel(
            "\n".join(header_lines),
            title="[bold blue]🔍 Synapse Trace[/]",
            border_style="blue",
        ))

        if dag_nodes_raw:
            import json as _json
            try:
                dag_nodes = _json.loads(dag_nodes_raw) if isinstance(dag_nodes_raw, str) else dag_nodes_raw
                if dag_nodes:
                    dag_table = Table(title="Task DAG", show_lines=False, box=None, padding=(0, 2))
                    dag_table.add_column("Node", style="dim", width=4, justify="right")
                    dag_table.add_column("Agent", style="cyan", width=12)
                    dag_table.add_column("Sub-agent", style="green", width=14)
                    dag_table.add_column("Instruction", style="white")
                    for i, node in enumerate(dag_nodes, 1):
                        dag_table.add_row(
                            str(i),
                            node.get("agent", "?"),
                            node.get("metadata", {}).get("sub_agent", "—"),
                            (node.get("instruction") or "")[:80],
                        )
                    console.print(dag_table)
            except Exception:
                pass

        tl_table = Table(
            title="Event Timeline",
            show_lines=False,
            box=None,
            padding=(0, 2),
        )
        tl_table.add_column("", style="dim", width=2)
        tl_table.add_column("Δt", style="dim", width=9, justify="right")
        tl_table.add_column("Event", width=26)
        tl_table.add_column("Source → Target", style="dim", width=32)
        tl_table.add_column("Detail", style="white")

        t0 = t.started_at
        for event in real_events:
            offset = (
                f"+{(event.timestamp - t0).total_seconds() * 1000:.0f}ms"
                if t0 else "—"
            )
            icon = _EVENT_ICONS.get(event.event_type, "·")
            colour = _EVENT_COLOURS.get(event.event_type, "white")
            type_str = f"[{colour}]{event.event_type}[/]"

            src_tgt = event.source
            if event.target:
                src_tgt += f" → {event.target}"

            detail = ""
            if event.error:
                detail = f"[red]✗ {event.error[:80]}[/]"
            elif event.payload:
                p = event.payload
                if event.event_type == "web_search_round":
                    rnd = p.get("round", "?")
                    n_urls = p.get("urls_found", 0)
                    n_ext = p.get("pages_extracted", 0)
                    detail = f"[dim]round {rnd}:[/] {n_urls} URLs found, {n_ext} extracted"
                elif event.event_type == "web_depth_analysis":
                    n_fups = p.get("n_followups", 0)
                    rnd = p.get("round", "?")
                    detail = f"[dim]round {rnd}:[/] {n_fups} follow-up queries generated"
                elif event.event_type == "msr_decision":
                    should_ask = p.get("should_ask", False)
                    n_q = p.get("n_questions", 0)
                    detail = f"[cyan]ask={should_ask}[/] {n_q} questions"
                elif event.event_type == "msr_answers":
                    ctx = str(p.get("user_context", ""))[:90]
                    detail = f"[dim]steering:[/] {ctx}"
                elif event.event_type == "web_synthesis":
                    n_art = p.get("n_articles", 0)
                    mode = p.get("mode", "")
                    detail = f"{n_art} articles · [dim]{mode}[/]"
                else:
                    for key in ("template", "reasoning", "output_preview", "query",
                                "approach", "agents_used", "tasks_succeeded", "agent"):
                        if key in p:
                            val = str(p[key])
                            label = {
                                "template": "→",
                                "reasoning": "reason:",
                                "output_preview": "output:",
                                "query": "q:",
                                "approach": "plan:",
                                "agents_used": "agents:",
                                "tasks_succeeded": "ok:",
                                "agent": "agent:",
                            }.get(key, f"{key}:")
                            detail = f"[dim]{label}[/] {val[:90]}"
                            break

            tl_table.add_row(icon, offset, type_str, src_tgt, detail)

        console.print(tl_table)

        if verbose:
            _print_verbose_web_trace(real_events, t0)

        console.print(
            f"[dim]  {len(real_events)} events · "
            f"Run [bold]octane dag \"<query>\"[/bold] to preview routing before executing[/]"
        )
        if not verbose:
            web_evts = [e for e in real_events if e.event_type in (
                "web_search_round", "web_depth_analysis", "msr_decision",
                "msr_answers", "web_synthesis",
            )]
            if web_evts:
                console.print(
                    f"[dim]  {len(web_evts)} web events hidden — use [bold]-v[/bold] / [bold]--verbose[/bold] to see every URL and chunk[/]"
                )

    else:
        trace_ids = synapse.list_traces(limit=15)
        if not trace_ids:
            console.print("[yellow]No traces found. Run 'octane ask' first.[/]")
            return

        table = Table(title="Recent Traces  (~/.octane/traces/)", show_lines=False)
        table.add_column("Correlation ID", style="cyan")
        table.add_column("Started", style="dim", width=20)
        table.add_column("Events", justify="right", style="dim")
        table.add_column("Duration", style="yellow", justify="right")
        table.add_column("Agents", style="green")
        table.add_column("", justify="center", width=3)

        for cid in trace_ids:
            if cid == "preflight":
                continue
            t = synapse.get_trace(cid)
            started_str = t.started_at.strftime("%m-%d %H:%M:%S") if t.started_at else "?"
            visible_agents = [a for a in t.agents_used if a not in ("user", "osa", "osa.decomposer", "osa.evaluator")]
            table.add_row(
                t.correlation_id,
                started_str,
                str(len(t.events)),
                f"{t.total_duration_ms:.0f}ms",
                ", ".join(visible_agents) or "—",
                "✅" if t.success else "❌",
            )

        console.print(table)
        console.print("[dim]  octane trace <id>         — full event timeline[/]")
        console.print("[dim]  octane trace <id> -v      — verbose: every URL, chunk, reasoning[/]")
        console.print("[dim]  octane trace <id>         — partial IDs accepted (first 8 chars)[/]")


def _print_verbose_web_trace(events, t0) -> None:
    """Render a rich verbose breakdown of all web-agent events."""
    from rich.rule import Rule as RichRule

    web_rounds = [e for e in events if e.event_type == "web_search_round"]
    depth_events = [e for e in events if e.event_type == "web_depth_analysis"]
    msr_dec = next((e for e in events if e.event_type == "msr_decision"), None)
    msr_ans = next((e for e in events if e.event_type == "msr_answers"), None)
    synth_evt = next((e for e in events if e.event_type == "web_synthesis"), None)

    if not any([web_rounds, depth_events, msr_dec, synth_evt]):
        return

    console.print()
    console.print(RichRule("[bold cyan]🔬 Verbose Web Trace[/]", style="cyan"))

    for evt in web_rounds:
        p = evt.payload
        rnd = p.get("round", "?")
        sub = p.get("sub_agent", "")
        queries_list = p.get("queries", [])
        urls = p.get("urls", [])
        extracted_detail = p.get("extracted_detail", [])
        n_found = p.get("urls_found", len(urls))
        n_ext = p.get("pages_extracted", len(extracted_detail))

        offset_str = (
            f"+{(evt.timestamp - t0).total_seconds() * 1000:.0f}ms"
            if t0 else ""
        )

        console.print()
        console.print(
            f"[bold cyan]🌐 Round {rnd} Search[/]  [dim]{sub} · {offset_str}[/]  "
            f"[green]{n_found} URLs found[/] · [yellow]{n_ext} extracted[/]"
        )

        if queries_list:
            console.print(f"  [dim]Queries:[/]")
            for q in queries_list:
                console.print(f"    [dim]·[/] {q}")

        if urls:
            url_table = Table(show_header=True, show_lines=False, box=None, padding=(0, 2))
            url_table.add_column("#", style="dim", width=3, justify="right")
            url_table.add_column("URL", style="cyan")
            url_table.add_column("Status", style="dim", width=10)
            extracted_urls = {d.get("url", "") for d in extracted_detail}
            for i, url in enumerate(urls[:30], 1):
                status_str = "[green]extracted[/]" if url in extracted_urls else "[dim]skipped[/]"
                url_table.add_row(str(i), url[:100], status_str)
            if len(urls) > 30:
                url_table.add_row("…", f"[dim]+{len(urls) - 30} more[/]", "")
            console.print(url_table)

        if extracted_detail:
            ext_table = Table(
                title=f"Extracted Pages — Round {rnd}",
                show_lines=False, box=None, padding=(0, 2),
            )
            ext_table.add_column("#", style="dim", width=3, justify="right")
            ext_table.add_column("URL", style="cyan")
            ext_table.add_column("Method", style="yellow", width=12)
            ext_table.add_column("Chars", style="green", width=8, justify="right")
            ext_table.add_column("Words", style="dim", width=8, justify="right")
            for i, d in enumerate(extracted_detail, 1):
                ext_table.add_row(
                    str(i),
                    d.get("url", "?")[:90],
                    d.get("method", "?"),
                    str(d.get("chars", 0)),
                    str(d.get("words", 0)),
                )
            console.print(ext_table)

    for evt in depth_events:
        p = evt.payload
        rnd = p.get("round", "?")
        followups = p.get("followup_queries", [])
        uctx = p.get("user_context") or ""
        offset_str = (
            f"+{(evt.timestamp - t0).total_seconds() * 1000:.0f}ms"
            if t0 else ""
        )

        console.print()
        console.print(
            f"[bold blue]🔍 Depth Analysis — Round {rnd}[/]  [dim]{offset_str}[/]  "
            f"[cyan]{len(followups)} follow-up queries[/]"
        )
        if uctx:
            console.print(f"  [dim]Steering:[/] {uctx}")

        if followups:
            fup_table = Table(show_header=True, show_lines=False, box=None, padding=(0, 2))
            fup_table.add_column("#", style="dim", width=3, justify="right")
            fup_table.add_column("Query", style="cyan")
            fup_table.add_column("API", style="yellow", width=8)
            fup_table.add_column("Rationale", style="dim")
            for i, fup in enumerate(followups, 1):
                fup_table.add_row(
                    str(i),
                    fup.get("query", "?")[:90],
                    fup.get("api", "search"),
                    fup.get("rationale", "")[:60],
                )
            console.print(fup_table)

    if msr_dec:
        p = msr_dec.payload
        should_ask = p.get("should_ask", False)
        questions = p.get("questions", [])
        offset_str = (
            f"+{(msr_dec.timestamp - t0).total_seconds() * 1000:.0f}ms"
            if t0 else ""
        )
        console.print()
        decision_str = "[green]asked clarification[/]" if should_ask else "[dim]skipped (query clear)[/]"
        console.print(f"[bold yellow]❓ MSR Decision  [dim]{offset_str}[/][/]  {decision_str}")
        if questions:
            for q_text in questions:
                console.print(f"  [dim]·[/] {q_text}")
        if msr_ans:
            ctx = msr_ans.payload.get("user_context", "")
            console.print(f"  [dim]User answered:[/] [cyan]{ctx}[/]")

    if synth_evt:
        p = synth_evt.payload
        n_art = p.get("n_articles", 0)
        deep = p.get("deep", False)
        mode = p.get("mode", "")
        offset_str = (
            f"+{(synth_evt.timestamp - t0).total_seconds() * 1000:.0f}ms"
            if t0 else ""
        )
        console.print()
        tier_str = "[bold magenta]REASON tier[/] (8B)" if deep else "[dim]MID tier[/] (Qwen)"
        console.print(
            f"[bold magenta]📝 Synthesis  [dim]{offset_str}[/][/]  "
            f"{n_art} articles → {tier_str}  [dim]{mode}[/]"
        )

    console.print()


def _resolve_trace_id(synapse, partial_id: str) -> str | None:
    """Return a full correlation ID that starts with partial_id, or None."""
    t = synapse.get_trace(partial_id)
    if t.events:
        return partial_id
    all_ids = synapse.list_traces(limit=50)
    for cid in all_ids:
        if cid.startswith(partial_id):
            return cid
    return None


def _format_agent_tags(agents_used: list[str]) -> str:
    """Return a Rich markup string of coloured agent tags."""
    _AGENT_COLOURS = {
        "web": "cyan", "code": "yellow", "memory": "blue",
        "sysstat": "green", "pnl": "magenta",
    }
    parts = []
    for a in agents_used:
        if a in ("user", "osa", "osa.decomposer", "osa.evaluator"):
            continue
        c = _AGENT_COLOURS.get(a, "white")
        parts.append(f"[bold {c}]{a}[/]")
    return "  ".join(parts) if parts else "[dim]osa[/]"
