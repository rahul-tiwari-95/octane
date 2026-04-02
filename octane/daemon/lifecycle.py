"""Daemon Lifecycle Manager — start, stop, status, drain.

Handles the full lifecycle of the Octane daemon process:
    - start:  Fork to background, write PID file, initialize all subsystems
    - stop:   Signal the daemon, wait for graceful shutdown
    - status: Report PID, uptime, queue depth, connection health
    - drain:  Stop accepting new tasks, wait for running tasks to complete

PID file: ~/.octane/daemon.pid
Socket:   ~/.octane/octane.sock

The daemon is started as a background asyncio process.  On start:
    1. Create ~/.octane/ directory
    2. Clean up stale socket/PID from previous crash
    3. Write PID file
    4. Initialize DaemonState, PoolManager, ModelManager, DaemonQueue
    5. Start DaemonServer on Unix socket
    6. Start background tasks (health checks, idle model sweeps, aging)
    7. Serve until SIGTERM/SIGINT

On stop:
    1. Send SIGTERM to daemon PID
    2. Daemon enters drain mode (queue stops accepting)
    3. Wait for active tasks to complete
    4. Close all pools
    5. Remove socket and PID files
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import structlog

from octane.daemon.client import get_pid_path, get_socket_path, is_daemon_running
from octane.daemon.inference_proxy import InferenceProxy
from octane.daemon.model_manager import ModelManager
from octane.daemon.pool import PoolManager
from octane.daemon.queue import AGING_INTERVAL_SEC, DaemonQueue, Priority, QueueItem
from octane.daemon.server import DaemonServer
from octane.daemon.state import DaemonState, DaemonStatus

logger = structlog.get_logger().bind(component="daemon_lifecycle")


class DaemonLifecycle:
    """Manages the full lifecycle of the Octane daemon.

    This class is the entry point for `octane daemon start`.  It initializes
    all daemon subsystems, starts the Unix socket server, and runs the event
    loop until shutdown.

    Args:
        topology_name: Topology for pool sizing and model management.
        foreground:    If True, run in foreground (don't fork). For debugging.
    """

    def __init__(
        self,
        topology_name: str = "auto",
        foreground: bool = False,
    ) -> None:
        self._topology_name = topology_name
        self._foreground = foreground

        # Subsystems (initialized in _init_subsystems)
        self.state: DaemonState | None = None
        self.queue: DaemonQueue | None = None
        self.pool: PoolManager | None = None
        self.model_manager: ModelManager | None = None
        self.inference_proxy: InferenceProxy | None = None
        self.server: DaemonServer | None = None

        # Background tasks
        self._aging_task: asyncio.Task | None = None
        self._imessage_shadow = None

    def resolve_topology(self) -> str:
        """Resolve 'auto' to actual topology name."""
        if self._topology_name != "auto":
            return self._topology_name
        try:
            from octane.tools.topology import detect_topology
            return detect_topology()
        except Exception:
            return "balanced"

    async def _init_subsystems(self) -> None:
        """Initialize all daemon subsystems."""
        topo = self.resolve_topology()
        logger.info("daemon_init", topology=topo, pid=os.getpid())

        self.state = DaemonState(topology_name=topo)
        await self.state.set_status(DaemonStatus.STARTING)

        self.queue = DaemonQueue()
        self.pool = PoolManager(state=self.state, topology_name=topo)
        self.model_manager = ModelManager(
            state=self.state,
            topology_name=topo,
            bodega_client=None,  # Will be set after pool init
        )

        # Initialize connection pools
        try:
            bodega = await self.pool.get_bodega()
            self.model_manager.bodega = bodega

            # ── Inference Proxy: backpressure-aware gateway to Bodega ─────
            self.inference_proxy = InferenceProxy(bodega)

            # Register all models defined in the topology so the proxy's
            # per-model semaphores are created before any request arrives.
            from octane.tools.topology import get_topology, ModelTier
            topo_obj = get_topology(topo)
            for tier, cfg in topo_obj.models.items():
                if tier == ModelTier.EMBED:
                    continue  # EMBED is in-process, not Bodega
                self.inference_proxy.register_model(
                    cfg.model_id, cfg.max_concurrency,
                    model_path=cfg.model_path,
                )

            # Sync InferenceProxy with the LIVE Bodega state.
            # Models already loaded in Bodega before daemon start (e.g. user
            # manually loaded them, or a previous daemon run left them loaded)
            # get registered so they can receive inference traffic immediately.
            # This also removes stale topology entries for models that are NOT
            # actually loaded — preventing slots being held for phantom models.
            try:
                health = await bodega.health()
                live_models = {
                    m["id"]: m.get("type", "lm")
                    for m in health.get("models_detail", [])
                    if m.get("status") == "running"
                }
                # Register any live model not already in the proxy.
                for live_id, live_type in live_models.items():
                    canonical = self.inference_proxy._resolve(live_id)
                    if canonical is None:
                        # Determine a sensible concurrency from topology or default.
                        default_conc = 2
                        for tier, cfg in topo_obj.models.items():
                            if (cfg.model_id.lower() == live_id.lower()
                                    or cfg.model_path.lower() == live_id.lower()):
                                default_conc = cfg.max_concurrency
                                break
                        self.inference_proxy.register_model(
                            live_id, default_conc, model_path=live_id,
                        )
                        logger.info(
                            "proxy_sync_registered_live_model",
                            model_id=live_id,
                            concurrency=default_conc,
                        )
                # Remove topology slots whose models are NOT in live Bodega.
                # Keeps the proxy accurate so `octane daemon models` reflects reality.
                topo_ids: set[str] = set()
                for tier, cfg in topo_obj.models.items():
                    if tier != ModelTier.EMBED:
                        topo_ids.add(cfg.model_id)
                for slot_id in list(self.inference_proxy._slots.keys()):
                    # Only remove topo-registered slots that aren't actually loaded.
                    if slot_id not in topo_ids:
                        continue
                    canonical_live = any(
                        slot_id.lower() == lid.lower()
                        or self.inference_proxy._resolve(lid) == slot_id
                        for lid in live_models
                    )
                    if not canonical_live:
                        self.inference_proxy.unregister_model(slot_id)
                        logger.info(
                            "proxy_sync_removed_phantom_model",
                            model_id=slot_id,
                        )
            except Exception as exc:
                logger.warning("proxy_sync_failed", error=str(exc))

            # Auto-load CLASSIFY model (vertex-4b) — daemon's private brain.
            classify_cfg = topo_obj.models.get(ModelTier.CLASSIFY)
            if classify_cfg:
                self.inference_proxy.classify_model = classify_cfg.model_id
                logger.info(
                    "classify_model_loading",
                    model_id=classify_cfg.model_id,
                    model_path=classify_cfg.model_path,
                )
                try:
                    params = classify_cfg.to_load_params()
                    await bodega.load_model(**params)
                    await self.model_manager.register_loaded(
                        classify_cfg.model_id, "classify",
                        classify_cfg.context_length * 0.002,  # rough VRAM
                    )
                    logger.info(
                        "classify_model_loaded",
                        model_id=classify_cfg.model_id,
                    )
                except Exception as exc:
                    # 409 Conflict = model already loaded in Bodega — treat as success.
                    if "409" in str(exc):
                        logger.info(
                            "classify_model_already_loaded",
                            model_id=classify_cfg.model_id,
                        )
                    else:
                        logger.warning(
                            "classify_model_load_failed",
                            model_id=classify_cfg.model_id,
                            error=str(exc),
                        )

        except Exception as exc:
            logger.warning("bodega_init_failed", error=str(exc))

        # Initialize Redis and Postgres pools
        try:
            await self.pool.get_redis()
        except Exception as exc:
            logger.warning("redis_init_failed", error=str(exc))

        try:
            await self.pool.get_postgres()
        except Exception as exc:
            logger.warning("postgres_init_failed", error=str(exc))

        # Set up server with request handler
        self.server = DaemonServer()
        self.server.set_handler(self._handle_request)

    async def _handle_request(self, command: str, payload: dict[str, Any]) -> Any:
        """Route incoming requests to appropriate handlers.

        This is the dispatcher — every request from the CLI comes through here.
        """
        handlers = {
            "status": self._handle_status,
            "health": self._handle_health,
            "snapshot": self._handle_snapshot,
            "ping": self._handle_ping,
            "drain": self._handle_drain,
            "pause_request": self._handle_pause_request,
            "resume_request": self._handle_resume_request,
            "list_requests": self._handle_list_requests,            # ── Inference gateway ─────────────────────────────────────────
            "infer": self._handle_infer,
            "models": self._handle_models,
            "load_model": self._handle_load_model,
            "unload_model": self._handle_unload_model,
            "pressure": self._handle_pressure,            # ── Queued commands ───────────────────────────────────────────────
            "ask": self._handle_ask,
            "investigate": self._handle_investigate,
            "compare": self._handle_compare,
        }

        handler = handlers.get(command)
        if handler is None:
            return {
                "status": "error",
                "error": f"Unknown command: {command}",
            }

        return await handler(payload)

    async def _handle_status(self, _payload: dict) -> dict:
        """Return daemon status."""
        snap = await self.state.snapshot() if self.state else {}
        queue_snap = self.queue.snapshot() if self.queue else {}
        return {
            "status": "ok",
            "data": {
                "daemon": snap,
                "queue": queue_snap,
                "server": self.server.snapshot() if self.server else {},
                "pools": self.pool.snapshot() if self.pool else {},
                "model_manager": self.model_manager.snapshot() if self.model_manager else {},
                "inference_proxy": self.inference_proxy.snapshot() if self.inference_proxy else {},
            },
        }

    async def _handle_health(self, _payload: dict) -> dict:
        """Quick health check."""
        return {
            "status": "ok",
            "data": {
                "daemon_status": self.state.status.value if self.state else "unknown",
                "uptime_seconds": round(self.state.uptime_seconds, 1) if self.state else 0,
                "queue_depth": self.queue.size if self.queue else 0,
                "active_connections": self.server.active_connections if self.server else 0,
            },
        }

    async def _handle_snapshot(self, _payload: dict) -> dict:
        """Full system snapshot."""
        return await self._handle_status(_payload)

    async def _handle_ping(self, _payload: dict) -> dict:
        """Simple ping for connectivity check."""
        return {"status": "ok", "data": {"pong": True}}

    async def _handle_drain(self, _payload: dict) -> dict:
        """Begin graceful drain."""
        asyncio.create_task(self._shutdown())
        return {"status": "ok", "data": {"draining": True}}

    async def _handle_list_requests(self, _payload: dict) -> dict:
        """List all pending and active queue items."""
        items = []
        if self.queue:
            for _pri, _seq, item in self.queue._items:
                items.append({
                    "task_id": item.task_id,
                    "command": item.command,
                    "priority": item.priority.name,
                    "wait_sec": round(item.wait_time, 1),
                    "aged_count": item.aged_count,
                    "paused": getattr(item, "paused", False),
                })
        return {"status": "ok", "data": {"requests": items}}

    async def _handle_pause_request(self, payload: dict) -> dict:
        """Pause a queued request by task_id (preserves position and context)."""
        task_id = payload.get("task_id", "")
        if not task_id:
            return {"status": "error", "error": "task_id required"}
        if self.queue:
            for _pri, _seq, item in self.queue._items:
                if item.task_id == task_id:
                    item.paused = True  # type: ignore[attr-defined]
                    logger.info("request_paused", task_id=task_id)
                    return {"status": "ok", "data": {"task_id": task_id, "paused": True}}
        return {"status": "error", "error": f"task_id not found: {task_id}"}

    async def _handle_resume_request(self, payload: dict) -> dict:
        """Resume a paused request by task_id."""
        task_id = payload.get("task_id", "")
        if not task_id:
            return {"status": "error", "error": "task_id required"}
        if self.queue:
            for _pri, _seq, item in self.queue._items:
                if item.task_id == task_id:
                    item.paused = False  # type: ignore[attr-defined]
                    logger.info("request_resumed", task_id=task_id)
                    return {"status": "ok", "data": {"task_id": task_id, "paused": False}}
        return {"status": "error", "error": f"task_id not found: {task_id}"}

    # ── Inference gateway handlers ────────────────────────────────────────────

    async def _handle_infer(self, payload: dict) -> dict:
        """Proxy an inference request through the InferenceProxy.

        The CLI sends:
            {"model": "axe-stealth-37b", "messages": [...], "temperature": 0.7,
             "max_tokens": 2048, "timeout": 300}

        The proxy acquires a per-model semaphore slot, calls Bodega, and
        returns the full chat completion response.
        """
        if not self.inference_proxy:
            return {"status": "error", "error": "inference proxy not initialized"}

        model = payload.get("model", "current")
        messages = payload.get("messages")
        if not messages:
            return {"status": "error", "error": "messages required"}

        temperature = payload.get("temperature", 0.7)
        max_tokens = payload.get("max_tokens", 2048)
        timeout = payload.get("timeout", 300.0)

        try:
            result = await self.inference_proxy.chat(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return {"status": "ok", "data": result}
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "error": f"inference slot timeout for {model} (waited {timeout}s)",
            }
        except Exception as exc:
            logger.error("infer_error", model=model, error=str(exc))
            return {"status": "error", "error": str(exc)}

    async def _handle_models(self, _payload: dict) -> dict:
        """List all models known to the InferenceProxy with pressure status."""
        if not self.inference_proxy:
            return {"status": "ok", "data": {"models": {}, "pressure": {}}}
        return {
            "status": "ok",
            "data": {
                "models": self.inference_proxy.snapshot()["models"],
                "pressure": self.inference_proxy.pressure_report(),
                "classify_model": self.inference_proxy.classify_model,
            },
        }

    async def _handle_load_model(self, payload: dict) -> dict:
        """Load a model via Bodega admin API and register in proxy."""
        if not self.inference_proxy or not self.inference_proxy.bodega:
            return {"status": "error", "error": "bodega not available"}

        model_path = payload.get("model_path")
        if not model_path:
            return {"status": "error", "error": "model_path required"}

        model_id = payload.get("model_id", model_path.split("/")[-1])
        max_concurrency = payload.get("max_concurrency", 1)

        try:
            result = await self.inference_proxy.bodega.load_model(**payload)
            self.inference_proxy.register_model(model_id, max_concurrency)
            if self.model_manager:
                await self.model_manager.register_loaded(
                    model_id,
                    payload.get("tier", "mid"),
                )
            return {"status": "ok", "data": result}
        except Exception as exc:
            logger.error("load_model_error", model_id=model_id, error=str(exc))
            return {"status": "error", "error": str(exc)}

    async def _handle_unload_model(self, payload: dict) -> dict:
        """Unload a model via Bodega admin API and deregister from proxy."""
        if not self.inference_proxy or not self.inference_proxy.bodega:
            return {"status": "error", "error": "bodega not available"}

        model_id = payload.get("model_id")
        if not model_id:
            return {"status": "error", "error": "model_id required"}

        # Prevent unloading the CLASSIFY model
        if model_id == self.inference_proxy.classify_model:
            return {
                "status": "error",
                "error": f"{model_id} is the daemon-exclusive CLASSIFY model and cannot be unloaded",
            }

        try:
            result = await self.inference_proxy.bodega.unload_model(model_id)
            self.inference_proxy.unregister_model(model_id)
            if self.model_manager:
                await self._unload_via_manager(model_id)
            return {"status": "ok", "data": result}
        except Exception as exc:
            logger.error("unload_model_error", model_id=model_id, error=str(exc))
            return {"status": "error", "error": str(exc)}

    async def _unload_via_manager(self, model_id: str) -> None:
        """Deregister from model_manager state (bypass Bodega call — already done)."""
        if self.state:
            await self.state.unregister_model(model_id)

    async def _handle_pressure(self, _payload: dict) -> dict:
        """Return per-model pressure report."""
        if not self.inference_proxy:
            return {"status": "ok", "data": {}}
        return {
            "status": "ok",
            "data": self.inference_proxy.pressure_report(),
        }

    # ── Queued command handlers ───────────────────────────────────────────────

    async def _handle_ask(self, payload: dict) -> Any:
        """Handle 'ask' command — runs through OSA with streaming response."""
        query = payload.get("query", "")
        if not query:
            return {"status": "error", "error": "query required"}

        # Create queue item for tracking
        task_id = str(uuid.uuid4())
        item = QueueItem(
            task_id=task_id,
            command="ask",
            payload=payload,
            priority=Priority.P0_INTERACTIVE,
        )
        await self.queue.submit(item)
        logger.info("ask_queued", task_id=task_id, query=query[:50])

        # Execute and stream results
        async def stream_ask():
            try:
                from octane.models.synapse import SynapseEventBus
                from octane.osa.orchestrator import Orchestrator

                synapse = SynapseEventBus()
                osa = Orchestrator(synapse)

                yield {"status": "stream", "data": {"task_id": task_id, "state": "running"}}

                result_parts = []
                async for chunk in osa.run_stream(query):
                    if isinstance(chunk, str):
                        result_parts.append(chunk)
                        yield {"status": "stream", "chunk": chunk}
                    elif hasattr(chunk, "text"):
                        result_parts.append(chunk.text)
                        yield {"status": "stream", "chunk": chunk.text}

                # Remove from queue
                await self.queue.remove(task_id)
                yield {
                    "status": "done",
                    "data": {
                        "task_id": task_id,
                        "result": "".join(result_parts),
                    },
                }
            except Exception as exc:
                logger.error("ask_error", task_id=task_id, error=str(exc))
                await self.queue.remove(task_id)
                yield {"status": "error", "error": str(exc), "task_id": task_id}

        return stream_ask()

    async def _handle_investigate(self, payload: dict) -> Any:
        """Handle 'investigate' command — multi-query deep research."""
        query = payload.get("query", "")
        if not query:
            return {"status": "error", "error": "query required"}

        task_id = str(uuid.uuid4())
        item = QueueItem(
            task_id=task_id,
            command="investigate",
            payload=payload,
            priority=Priority.P0_INTERACTIVE,
        )
        await self.queue.submit(item)
        logger.info("investigate_queued", task_id=task_id, query=query[:50])

        async def stream_investigate():
            try:
                from octane.osa.investigate import InvestigateOrchestrator
                from octane.agents.web.agent import WebAgent
                from octane.models.synapse import SynapseEventBus

                synapse = SynapseEventBus()
                web_agent = WebAgent(synapse)
                orchestrator = InvestigateOrchestrator(web_agent=web_agent)

                max_dimensions = payload.get("max_dimensions")
                kwargs = {}
                if max_dimensions is not None:
                    kwargs["max_dimensions"] = max_dimensions

                yield {"status": "stream", "data": {"task_id": task_id, "state": "running"}}

                async for event in orchestrator.run_stream(query, **kwargs):
                    yield {"status": "stream", "data": event}

                await self.queue.remove(task_id)
                yield {"status": "done", "data": {"task_id": task_id}}
            except Exception as exc:
                logger.error("investigate_error", task_id=task_id, error=str(exc))
                await self.queue.remove(task_id)
                yield {"status": "error", "error": str(exc), "task_id": task_id}

        return stream_investigate()

    async def _handle_compare(self, payload: dict) -> Any:
        """Handle 'compare' command — side-by-side comparison research."""
        query = payload.get("query", "")
        if not query:
            return {"status": "error", "error": "query required"}

        task_id = str(uuid.uuid4())
        item = QueueItem(
            task_id=task_id,
            command="compare",
            payload=payload,
            priority=Priority.P0_INTERACTIVE,
        )
        await self.queue.submit(item)
        logger.info("compare_queued", task_id=task_id, query=query[:50])

        async def stream_compare():
            try:
                from octane.osa.compare import CompareOrchestrator
                from octane.agents.web.agent import WebAgent
                from octane.models.synapse import SynapseEventBus

                synapse = SynapseEventBus()
                web_agent = WebAgent(synapse)
                orchestrator = CompareOrchestrator(web_agent=web_agent)

                yield {"status": "stream", "data": {"task_id": task_id, "state": "running"}}

                async for event in orchestrator.run_stream(query):
                    yield {"status": "stream", "data": event}

                await self.queue.remove(task_id)
                yield {"status": "done", "data": {"task_id": task_id}}
            except Exception as exc:
                logger.error("compare_error", task_id=task_id, error=str(exc))
                await self.queue.remove(task_id)
                yield {"status": "error", "error": str(exc), "task_id": task_id}

        return stream_compare()

    async def run(self) -> None:
        """Run the daemon — initialize, serve, shutdown.

        This is the main entry point. It blocks until the daemon is
        stopped via SIGTERM/SIGINT.
        """
        pid_path = get_pid_path()
        socket_path = get_socket_path()

        try:
            # Ensure ~/.octane/ exists
            socket_path.parent.mkdir(parents=True, exist_ok=True)

            # Write PID file
            pid_path.write_text(str(os.getpid()))

            # Initialize subsystems
            await self._init_subsystems()

            # Start server
            await self.server.start()

            # Start background tasks
            await self.pool.start_health_checks()
            await self.model_manager.start()
            self._aging_task = asyncio.create_task(self._aging_loop())

            # Start iMessage shadow if configured
            await self._start_imessage_shadow()

            # Mark as running
            await self.state.set_status(DaemonStatus.RUNNING)

            logger.info(
                "daemon_running",
                pid=os.getpid(),
                socket=str(socket_path),
                topology=self.state.topology_name,
            )

            # Install signal handlers
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self._shutdown()))

            # Run forever until shutdown
            stop_event = asyncio.Event()

            # Store stop event so _shutdown can set it
            self._stop_event = stop_event
            await stop_event.wait()

        except Exception as exc:
            logger.error("daemon_fatal_error", error=str(exc))
        finally:
            await self._cleanup(pid_path)

    async def _shutdown(self) -> None:
        """Graceful shutdown sequence."""
        logger.info("daemon_shutdown_initiated")

        if self.state:
            await self.state.set_status(DaemonStatus.DRAINING)

        # Stop accepting new tasks
        if self.queue:
            await self.queue.drain()

        # Wait for queue to empty (max 30s)
        if self.queue and not self.queue.is_empty:
            logger.info("daemon_draining_queue", remaining=self.queue.size)
            deadline = time.monotonic() + 30.0
            while not self.queue.is_empty and time.monotonic() < deadline:
                await asyncio.sleep(0.5)

        if self.state:
            await self.state.set_status(DaemonStatus.STOPPING)

        # Stop background tasks
        if self._imessage_shadow:
            try:
                await self._imessage_shadow.stop()
                logger.info("imessage_shadow_stopped")
            except Exception as exc:
                logger.warning("imessage_shadow_stop_error", error=str(exc))

        if self._aging_task and not self._aging_task.done():
            self._aging_task.cancel()
            try:
                await self._aging_task
            except asyncio.CancelledError:
                pass

        # Stop model manager
        if self.model_manager:
            await self.model_manager.stop()

        # Stop server
        if self.server:
            await self.server.stop()

        # Close pools
        if self.pool:
            await self.pool.close_all()

        if self.state:
            await self.state.set_status(DaemonStatus.STOPPED)

        # Signal the run() method to exit
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

        logger.info("daemon_shutdown_complete")

    async def _cleanup(self, pid_path: Path) -> None:
        """Clean up PID and socket files."""
        try:
            pid_path.unlink(missing_ok=True)
        except OSError:
            pass

    async def _aging_loop(self) -> None:
        """Background task: periodic priority aging for queued tasks."""

    async def _start_imessage_shadow(self) -> None:
        """Start iMessage shadow if config exists and is enabled."""
        import json as _json

        config_file = Path.home() / ".octane" / "macos" / "imessage" / "config.json"
        if not config_file.exists():
            return

        try:
            cfg = _json.loads(config_file.read_text())
        except (ValueError, OSError):
            return

        if not cfg.get("enabled"):
            return

        contacts = cfg.get("approved_contacts", [])
        if not contacts:
            return

        try:
            from octane.macos.imessage_shadow import IMessageShadow

            shadow = IMessageShadow(
                approved_contacts=contacts,
                allow_self=cfg.get("allow_self", False),
                poll_interval=cfg.get("poll_interval", 5.0),
            )
            await shadow.start()
            self._imessage_shadow = shadow
            logger.info(
                "imessage_shadow_started",
                contacts=len(contacts),
                allow_self=cfg.get("allow_self", False),
            )
        except Exception as exc:
            logger.warning("imessage_shadow_start_failed", error=str(exc))
        try:
            while True:
                await asyncio.sleep(AGING_INTERVAL_SEC)
                if self.queue:
                    aged = await self.queue.age_items()
                    if aged > 0:
                        logger.debug("queue_aging_sweep", aged_count=aged)
        except asyncio.CancelledError:
            pass


# ── CLI-facing functions ──────────────────────────────────────────────────────


async def start_daemon(
    topology: str = "auto",
    foreground: bool = False,
) -> bool:
    """Start the daemon.

    If foreground=True, runs in the current process (for debugging).
    Otherwise, forks to background.

    Returns True if started successfully.
    """
    if is_daemon_running():
        logger.info("daemon_already_running")
        return False

    lifecycle = DaemonLifecycle(topology_name=topology, foreground=foreground)

    if foreground:
        await lifecycle.run()
        return True

    # Fork to background
    pid = os.fork()
    if pid > 0:
        # Parent — wait briefly to check if child started OK
        await asyncio.sleep(1.0)
        if is_daemon_running():
            return True
        return False
    else:
        # Child — become session leader, run daemon with fresh event loop
        os.setsid()
        try:
            # Create a new event loop for the forked child process
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.run(lifecycle.run())
        except Exception as exc:
            logger.error("daemon_child_failed", error=str(exc))
        finally:
            os._exit(0)

    return True  # Unreachable, but keeps type checker happy


async def stop_daemon(timeout: float = 30.0) -> bool:
    """Stop the running daemon.

    Sends SIGTERM and waits for the process to exit.
    Returns True if stopped successfully.
    """
    pid_path = get_pid_path()

    if not is_daemon_running():
        logger.info("daemon_not_running")
        return False

    pid = int(pid_path.read_text().strip())

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        # Already dead
        _cleanup_files()
        return True

    # Wait for process to exit
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)  # Check if still alive
        except ProcessLookupError:
            _cleanup_files()
            return True
        await asyncio.sleep(0.5)

    # Still alive after timeout — force kill
    try:
        os.kill(pid, signal.SIGKILL)
        await asyncio.sleep(1.0)
    except ProcessLookupError:
        pass

    _cleanup_files()
    return True


async def daemon_status() -> dict[str, Any]:
    """Get daemon status.

    Returns a dict with status info, or indicates daemon is not running.
    """
    if not is_daemon_running():
        return {
            "running": False,
            "message": "Daemon is not running",
        }

    # Connect to daemon and request status
    from octane.daemon.client import DaemonClient
    client = DaemonClient()
    try:
        if not await client.connect(timeout=3.0):
            return {
                "running": True,
                "message": "Daemon running but not responding",
                "pid": get_pid_path().read_text().strip() if get_pid_path().exists() else "?",
            }

        response = await client.request("status", {}, timeout=5.0)
        response["running"] = True
        return response
    except Exception as exc:
        return {
            "running": True,
            "message": f"Error querying daemon: {exc}",
        }
    finally:
        await client.close()


def _cleanup_files() -> None:
    """Remove socket and PID files."""
    for path in (get_socket_path(), get_pid_path()):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
