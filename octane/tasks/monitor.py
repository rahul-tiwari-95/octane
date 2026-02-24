"""monitor_ticker — perpetual background task.

Runs every ``POLL_INTERVAL`` (default: 1 hour).  On each execution it:
  1. Calls the Bodega Intelligence finance endpoint for the requested ticker.
  2. Stores the latest quote in Redis under ``watch:<ticker>:latest``.
  3. Appends a timestamped snapshot to ``watch:<ticker>:history`` (capped at 48
     entries — 2 days at hourly cadence).
  4. Publishes a Redis pub/sub notification on ``watch:<ticker>:events``.

The task is *not* ``automatic=True`` by default — it is scheduled explicitly by
``octane watch <ticker>`` so that each ticker gets its own perpetual loop with a
stable key == ticker symbol.

Design constraints
------------------
- No Octane-internal imports at module top level to keep serialization clean
  (cloudpickle sends the function object across Redis; top-level imports that
  pull in heavy Octane state can silently break deserialization in the worker
  subprocess).
- ``BodegaIntelClient`` and ``RedisClient`` are instantiated *inside* the task
  body so the worker subprocess creates its own connections.
- Fallback: if Bodega is unreachable the task logs a warning and perpetuates
  normally — it does NOT crash and will retry on the next cycle.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from logging import LoggerAdapter, Logger

from shadows import Perpetual
from shadows.dependencies import TaskLogger

logger = logging.getLogger(__name__)

# How often the monitor polls Bodega Intelligence
POLL_INTERVAL = timedelta(hours=1)

# Maximum number of historical snapshots kept per ticker
MAX_HISTORY = 48


async def monitor_ticker(
    ticker: str,
    perpetual: Perpetual = Perpetual(every=POLL_INTERVAL),
    log: LoggerAdapter[Logger] = TaskLogger(),
) -> None:
    """Perpetual background task: poll Bodega Finance for *ticker* every hour.

    Schedule via::

        await shadow.add(monitor_ticker, key=ticker)(ticker=ticker)

    The ticker key ensures only one perpetual loop per symbol runs at a time
    (Shadows deduplicates by key).

    Args:
        ticker: Stock / crypto symbol (e.g. ``"AAPL"``, ``"BTC"``)
        perpetual: Injected by Shadows — controls re-scheduling cadence.
        log: Injected by Shadows — context-aware task logger.
    """
    # --- imports inside body to stay serialization-safe ----------------
    from octane.tools.bodega_intel import BodegaIntelClient
    from octane.config import settings

    now = datetime.now(timezone.utc)
    log.info("monitor_ticker: polling %s at %s", ticker, now.isoformat())

    quote: dict = {}
    try:
        async with BodegaIntelClient(settings.bodega_intel_url) as intel:
            result = await intel.finance(ticker)
            # BodegaIntelClient returns a dict or raises on failure
            quote = result if isinstance(result, dict) else {"raw": str(result)}
    except Exception as exc:
        log.warning("monitor_ticker: Bodega unreachable for %s — %s", ticker, exc)
        # Perpetuate normally; we will try again next cycle
        return

    # Enrich with a timestamp so Redis history is human-readable
    snapshot: dict = {
        "ticker": ticker.upper(),
        "ts": now.isoformat(),
        **quote,
    }
    serialised = json.dumps(snapshot)

    # Use redis.asyncio directly for list + pubsub operations not in RedisClient
    try:
        import redis.asyncio as aioredis  # type: ignore

        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        try:
            latest_key = f"watch:{ticker.upper()}:latest"
            history_key = f"watch:{ticker.upper()}:history"
            events_channel = f"watch:{ticker.upper()}:events"

            async with r.pipeline() as pipe:
                pipe.set(latest_key, serialised)
                pipe.rpush(history_key, serialised)
                pipe.ltrim(history_key, -MAX_HISTORY, -1)
                await pipe.execute()

            # Pub/sub notification (best-effort)
            try:
                await r.publish(events_channel, serialised)
            except Exception:
                pass

            log.info(
                "monitor_ticker: stored quote for %s — price=%s",
                ticker,
                quote.get("price") or quote.get("c") or "?",
            )
        finally:
            await r.aclose()
    except Exception as exc:
        log.warning("monitor_ticker: Redis write failed for %s — %s", ticker, exc)
