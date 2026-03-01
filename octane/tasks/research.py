"""research_cycle — Shadows perpetual task for long-running research.

Each execution:
  1. Logs cycle start to Redis ring buffer.
  2. Increments the cycle counter.
  3. Runs the full OSA pipeline (guard → decompose → web search + content
     extraction → synthesis) using Orchestrator.run_stream().
  4. Streams Synapse events to the Redis log in real time so --follow shows
     in-depth pipeline activity.
  5. Stores the synthesised output as a ResearchFinding in Postgres.
  6. Logs cycle completion with word count.

Scheduling:
    Registered as a Shadows perpetual task with ``every=RESEARCH_INTERVAL``
    (default 6 hours). The Shadows Worker re-executes it automatically.

Serialization safety:
    All Octane imports are deferred inside the function body — same pattern
    as ``monitor_ticker`` — so cloudpickle serializes only the code object.

Usage::

    from shadows import Shadow
    async with Shadow(name="octane", url=redis_url) as shadow:
        shadow.register(research_cycle)
        await shadow.add(research_cycle, key=task_id)(
            task_id=task_id, topic=topic, interval_hours=interval_hours
        )
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from logging import Logger, LoggerAdapter

from shadows import Perpetual
from shadows.dependencies import TaskLogger

logger = logging.getLogger(__name__)

# Default research cadence — overridable per-call but Shadows uses the default
# for its internal scheduling when the parameter is not varied.
RESEARCH_INTERVAL = timedelta(hours=6)

## TODO Why is hours hardcoded as 6 hours?
## TODO This is a reasonable default based on typical research cycles, but it should be configurable.

# Minimum word count for a finding to be stored.  Cycles that produce fewer
# words are almost certainly Bodega-down junk (identical keyword fallbacks)
# and should be discarded rather than polluting the findings table.
MINIMUM_QUALITY_WORDS = 30

# ── Event → human label ───────────────────────────────────────────────────────

def _event_label(event) -> str | None:  # noqa: ANN001
    """Convert a SynapseEvent into a short log line for --follow display.

    Returns None for noisy low-value events to avoid flooding the log.
    """
    t = event.event_type
    p = event.payload or {}
    src = event.source or ""

    if t == "ingress":
        q = p.get("query", "")[:100]
        return f"  → ingress: {q}"
    if t == "guard_block":
        return f"  🛡 guard blocked: {p.get('reason', '')[:80]}"
    if t == "decomposition_start":
        return f"  🗂 decomposing query…"
    if t == "decomposition_complete":
        nodes = p.get("nodes", p.get("node_count", "?"))
        return f"  🗂 DAG ready — {nodes} task(s)"
    if t == "dispatch":
        agent = p.get("agent", src)
        task_label = p.get("task", "")[:60]
        return f"  🚀 dispatch → {agent}: {task_label}"
    if t == "agent_start":
        return f"  ▶ {src} started"
    if t == "agent_complete":
        ms = int(event.duration_ms or 0)
        return f"  ✔ {src} done ({ms}ms)"
    if t == "agent_error":
        return f"  ✖ {src} error: {event.error[:80]}"
    if t in ("code_success", "code_healed"):
        attempt = p.get("attempt", 1)
        healed = "self-healed " if t == "code_healed" else ""
        return f"  💻 code {healed}✅ (attempt {attempt})"
    if t == "code_attempt_failed":
        attempt = p.get("attempt", "?")
        err = p.get("error_summary", "")[:60]
        return f"  💻 code ✗ attempt {attempt}: {err}"
    if t == "code_debug_invoked":
        return f"  🔧 debugger invoked (attempt {p.get('attempt', '?')})"
    if t == "code_exhausted":
        return f"  💻 code exhausted all retries"
    if t == "catalyst_success":
        return f"  ⚡ catalyst matched — skipping LLM"
    if t == "egress":
        # Prefer the exact word_count injected by run_stream/run; fall back to
        # counting tokens from output_preview (capped at 200 chars) or the
        # legacy "output"/"answer" keys used by run() and run_from_dag().
        words = (
            p.get("word_count")
            or len((p.get("output_preview") or p.get("output") or p.get("answer") or "").split())
        )
        agents = ", ".join(p.get("agents_used", []))
        return f"  ← egress: {words} words  [{agents}]"
    # Skip preflight / hil_complete / catalyst_failed (too noisy)
    return None


async def research_cycle(
    task_id: str,
    topic: str,
    interval_hours: float = 6.0,
    depth: str = "deep",
    perpetual: Perpetual = Perpetual(every=RESEARCH_INTERVAL),
    log: LoggerAdapter[Logger] = TaskLogger(),
) -> None:
    """Perpetual background research task.

    Schedule via::

        await shadow.add(research_cycle, key=task_id)(
            task_id=task_id, topic=topic, interval_hours=interval_hours,
            depth=depth,
        )

    Args:
        task_id:        Stable Shadows key — also used as research task identifier.
        topic:          Research topic / question (e.g. ``"NVDA earnings outlook"``).
        interval_hours: Target cycle cadence (metadata only; Shadows uses ``every``).
        depth:          One of ``"shallow"`` (2 angles), ``"deep"`` (4, default),
                        ``"exhaustive"`` (8). Controls breadth of sub-queries.
        perpetual:      Injected by Shadows — controls automatic re-scheduling.
        log:            Injected by Shadows — context-aware task logger.
    """
    # ── All imports deferred for cloudpickle safety ───────────────────────
    import asyncio as _asyncio

    from octane.config import settings
    from octane.models.synapse import SynapseEventBus, SynapseEvent
    from octane.osa.orchestrator import Orchestrator
    from octane.research.store import ResearchStore
    from octane.research.angles import AngleGenerator

    store = ResearchStore(
        redis_url=settings.redis_url,
        postgres_url=settings.postgres_url,
    )

    # ── 0. Guard: skip if task was stopped ────────────────────────────────
    task_meta = await store.get_task(task_id)
    if task_meta is not None and task_meta.status == "stopped":
        log.info("research_cycle: task stopped — skipping cycle  task=%s", task_id)
        await store.close()
        return

    # ── 1. Log cycle start ────────────────────────────────────────────────
    await store.log_entry(task_id, f"⚙ Cycle start — topic: {topic}  depth={depth}")
    cycle_num = await store.increment_cycle(task_id)
    log.info("research_cycle: start — task=%s cycle=%d topic=%s depth=%s",
             task_id, cycle_num, topic, depth)

    # ── 2. Build a logging-enabled Synapse bus ────────────────────────────
    class _ResearchSynapse(SynapseEventBus):
        """SynapseEventBus that also logs notable events to the research store."""

        def emit(self, event: SynapseEvent) -> None:  # type: ignore[override]
            super().emit(event)
            label = _event_label(event)
            if label:
                try:
                    _asyncio.ensure_future(store.log_entry(task_id, label))
                except RuntimeError:
                    pass

    synapse = _ResearchSynapse(persist=False)

    # ── 3. Generate multi-angle sub-queries ───────────────────────────────
    try:
        from octane.tools.bodega_router import BodegaRouter
        _bodega = BodegaRouter()
    except Exception:
        _bodega = None

    # ── 3a. Pre-cycle Bodega health check ─────────────────────────────────
    bodega_available = False
    if _bodega is not None:
        try:
            _health = await _bodega.health()
            bodega_available = _health.get("status") == "ok"
        except Exception:
            bodega_available = False
    if not bodega_available:
        await store.log_entry(task_id, "⚠ Bodega unavailable — cycle will use keyword fallbacks")
        log.warning("research_cycle: bodega unavailable  task=%s cycle=%d", task_id, cycle_num)

    angle_gen = AngleGenerator(bodega=_bodega)
    angles = await angle_gen.generate(topic, depth=depth)
    await store.log_entry(
        task_id,
        f"🔭 {len(angles)} angles: {', '.join(a['angle'] for a in angles)}",
    )
    log.info("research_cycle: angles — %d generated", len(angles))

    # ── 4. URL dedup set (per-task Redis SADD) ───────────────────────────
    # Key: research:urls:{task_id}  — SET of url hashes already fetched this task
    async def _is_url_seen(url: str) -> bool:
        """Return True if this URL was fetched in a previous cycle."""
        try:
            import hashlib
            r = await store._redis()
            if r is None:
                return False
            h = hashlib.sha256(url.lower().rstrip("/").encode()).hexdigest()
            return bool(await r.sismember(f"research:urls:{task_id}", h))
        except Exception:
            return False

    async def _mark_url_seen(url: str) -> None:
        try:
            import hashlib
            r = await store._redis()
            if r is None:
                return
            h = hashlib.sha256(url.lower().rstrip("/").encode()).hexdigest()
            await r.sadd(f"research:urls:{task_id}", h)
            await r.expire(f"research:urls:{task_id}", 60 * 60 * 24 * 30)  # 30-day TTL
        except Exception:
            pass

    # Cross-angle dedup within this cycle: shared in-memory set so two angles
    # that happen to fetch the same URL don't produce identical output.
    _seen_this_cycle: set[str] = set()

    # ── 5. Run parallel angle queries via OSA ────────────────────────────
    # Each angle gets its own Orchestrator instance to avoid shared-state
    # races when queries run concurrently.  Each query is also capped at
    # 90 s so a single hung web-agent can never block the whole cycle.
    output_parts: list[str] = []
    all_agents_used: list[str] = []

    async def _run_angle(angle: dict) -> str:
        q = angle["query"]
        label = angle["angle"]
        try:
            await store.log_entry(task_id, f"  🔎 [{label}] {q}")
            osa = Orchestrator(synapse, hil_interactive=False)
            # Skip the per-angle Bodega health-check preflight: it blocks for up to
            # 120 s when Bodega is busy generating a prior angle's synthesis response
            # (local MLX models are single-threaded).  Bodega reachability was already
            # verified when the AngleGenerator ran at cycle start.
            osa._preflight_done = True
            parts: list[str] = []

            async def _collect() -> str:
                nonlocal parts
                async for chunk in osa.run_stream(q, session_id=f"research_{task_id}_{label}"):
                    parts.append(chunk)
                return "".join(parts).strip()

            result = await _asyncio.wait_for(_collect(), timeout=300.0)

            # Cross-angle dedup: skip result if it's identical to one already
            # collected this cycle (happens when Bodega is down and all angles
            # fall back to the same keyword-extracted text).
            if result:
                import hashlib as _hl
                result_hash = _hl.sha256(result.encode()).hexdigest()
                if result_hash in _seen_this_cycle:
                    await store.log_entry(
                        task_id,
                        f"  🔁 [{label}] duplicate output — skipped",
                    )
                    return ""
                _seen_this_cycle.add(result_hash)

            return result
        except _asyncio.TimeoutError:
            await store.log_entry(task_id, f"  ⏱ [{label}] timed out after 300 s — skipped")
            return ""
        except Exception as exc:
            await store.log_entry(task_id, f"  ⚠ [{label}] error: {str(exc)[:80]}")
            return ""

    # Run angles sequentially — LLM evaluation is GPU-bound so parallel
    # requests contend on the same device and all timeout together.
    await store.log_entry(task_id, "⚙ Running angle queries…")
    try:
        angle_results = []
        for a in angles:
            angle_results.append(await _run_angle(a))
    except Exception as exc:
        err_msg = str(exc)[:120]
        log.warning("research_cycle: pipeline error — %s", err_msg)
        await store.log_entry(task_id, f"⚠ Pipeline error: {err_msg}")
        await store.close()
        return

    for r in angle_results:
        if isinstance(r, str) and r:
            output_parts.append(r)

    # Extract agents used from all synapse egress events
    for e in synapse._events:
        if e.event_type == "egress":
            all_agents_used.extend(e.payload.get("agents_used", []))
    all_agents_used = list(set(all_agents_used))

    content = "\n\n---\n\n".join(output_parts).strip()
    if not content:
        await store.log_entry(task_id, "⚠ All angles produced no output this cycle")
        await store.close()
        return

    # ── Quality gate ──────────────────────────────────────────────────────
    total_words = len(content.split())
    if total_words < MINIMUM_QUALITY_WORDS:
        await store.log_entry(
            task_id,
            f"⚠ Cycle {cycle_num} skipped — only {total_words} words "
            f"(minimum {MINIMUM_QUALITY_WORDS}).  "
            f"{'Bodega was unavailable; ' if not bodega_available else ''}"
            f"Junk output discarded.",
        )
        log.warning(
            "research_cycle: quality gate failed  task=%s cycle=%d words=%d",
            task_id, cycle_num, total_words,
        )
        await store.close()
        return

    # ── 6. Chunk & embed content for semantic search ──────────────────────
    try:
        from octane.tools.pg_client import PgClient
        from octane.tools.structured_store import EmbeddingEngine
        _pg = PgClient()
        await _pg.connect()
        if _pg.available:
            engine = EmbeddingEngine(_pg)
            n_chunks = await engine.embed_and_store(
                source_type="research_finding",
                source_id=cycle_num,
                text=content,
            )
            await store.log_entry(task_id, f"  🧠 {n_chunks} chunks embedded")
            await _pg.close()
    except Exception as _ee:
        log.warning("research_cycle: embed error — %s", str(_ee)[:80])

    # ── 7. Store finding ──────────────────────────────────────────────────
    finding = await store.add_finding(
        task_id=task_id,
        cycle_num=cycle_num,
        topic=topic,
        content=content,
        agents_used=all_agents_used,
        sources=[],
        word_count=len(content.split()),
    )

    if finding:
        await store.log_entry(
            task_id,
            f"✅ Cycle {cycle_num} complete — {finding.word_count} words stored  "
            f"({len(angles)} angles, depth={depth})",
        )
        log.info(
            "research_cycle: complete — task=%s cycle=%d words=%d depth=%s",
            task_id, cycle_num, finding.word_count, depth,
        )
    else:
        await store.log_entry(
            task_id,
            f"⚠ Cycle {cycle_num} — finding not stored (Postgres unavailable?)",
        )

    await store.close()

