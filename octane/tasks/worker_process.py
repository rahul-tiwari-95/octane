"""Octane Shadows Worker — subprocess entrypoint.

Launched by ``octane watch`` as a detached background process.
The process runs until killed and auto-schedules all ``automatic=True`` perpetual
tasks on startup.

Usage (internal — do not call directly)::

    python -m octane.tasks.worker_process \
        --shadows-name octane \
        --redis-url redis://localhost:6379/0

The worker registers all tasks from ``octane.tasks:octane_tasks`` so that any
task scheduled via Shadow (e.g. monitor_ticker) will be picked up.

PID file
--------
Written to ``~/.octane/worker.pid`` so that ``octane watch status`` and
``octane watch stop`` can find the process without needing a daemon manager.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

logger = logging.getLogger("octane.worker")

PID_DIR = Path.home() / ".octane"
PID_FILE = PID_DIR / "worker.pid"


def _write_pid() -> None:
    PID_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def read_pid() -> int | None:
    """Return the PID stored in the PID file, or None if absent / stale."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        # Check the process is actually alive
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        _remove_pid()
        return None


async def _run(shadows_name: str, redis_url: str) -> None:
    from shadows import Shadow, Worker

    _write_pid()
    logger.info("Octane Worker starting (PID=%d)", os.getpid())

    try:
        async with Shadow(name=shadows_name, url=redis_url) as shadow:
            shadow.register_collection("octane.tasks:octane_tasks")

            async with Worker(
                shadow,
                schedule_automatic_tasks=True,
            ) as worker:
                logger.info(
                    "Worker ready — registered tasks: %s",
                    list(shadow.tasks.keys()),
                )
                await worker.run_forever()
    finally:
        _remove_pid()
        logger.info("Octane Worker stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Octane background worker")
    parser.add_argument(
        "--shadows-name", default="octane", help="Shadows namespace (Redis key prefix)"
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis connection URL",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # Graceful shutdown on SIGTERM / SIGINT
    loop = asyncio.new_event_loop()

    def _shutdown(signum, frame):  # noqa: ARG001
        logger.info("Received signal %d — shutting down", signum)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        loop.run_until_complete(_run(args.shadows_name, args.redis_url))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
