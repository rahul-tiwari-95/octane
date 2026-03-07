"""Daemon shared state — the single source of truth while the daemon runs.

Tracks:
    - Topology configuration (which models, which tier mapping)
    - Active model registry (which models are loaded, last used time)
    - Session tracker (active sessions, their priorities)
    - Connection health (Redis, Postgres, Bodega alive?)
    - Daemon lifecycle (uptime, PID, status)

Thread-safe through asyncio locks.  Observable via change callbacks —
any component can register a listener for state changes.

Design: This is the "process control block" of the Octane daemon.
Like an OS kernel's task_struct, it holds everything the scheduler
and subsystems need to make decisions.
"""

from __future__ import annotations

import asyncio
import enum
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger().bind(component="daemon_state")


# ── Connection Health ─────────────────────────────────────────────────────────


class ConnectionStatus(str, enum.Enum):
    """Health status of an external dependency."""

    CONNECTED = "connected"       # Healthy, responding
    DEGRADED = "degraded"         # Responding but slow or erroring
    DISCONNECTED = "disconnected" # Unreachable
    UNKNOWN = "unknown"           # Not yet checked


@dataclass
class ConnectionHealth:
    """Health snapshot for one external service."""

    status: ConnectionStatus = ConnectionStatus.UNKNOWN
    last_check: float = 0.0           # Monotonic timestamp
    last_success: float = 0.0         # Monotonic timestamp of last successful check
    latency_ms: float = 0.0           # Last measured latency
    consecutive_failures: int = 0     # Failures since last success
    error: str = ""                   # Last error message

    def mark_healthy(self, latency_ms: float = 0.0) -> None:
        now = time.monotonic()
        self.status = ConnectionStatus.CONNECTED
        self.last_check = now
        self.last_success = now
        self.latency_ms = latency_ms
        self.consecutive_failures = 0
        self.error = ""

    def mark_failed(self, error: str = "") -> None:
        now = time.monotonic()
        self.consecutive_failures += 1
        self.last_check = now
        self.error = error
        # 3 consecutive failures → disconnected, else degraded
        if self.consecutive_failures >= 3:
            self.status = ConnectionStatus.DISCONNECTED
        else:
            self.status = ConnectionStatus.DEGRADED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 1),
            "consecutive_failures": self.consecutive_failures,
            "error": self.error,
        }


# ── Model Registry Entry ─────────────────────────────────────────────────────


@dataclass
class LoadedModel:
    """Tracks a model currently loaded in Bodega."""

    model_id: str                        # e.g. "bodega-raptor-8b"
    tier: str                            # "fast", "mid", "reason"
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    request_count: int = 0
    estimated_memory_mb: float = 0.0     # Approximate VRAM usage

    def touch(self) -> None:
        """Record a usage — updates last_used and increments count."""
        self.last_used = time.monotonic()
        self.request_count += 1

    @property
    def idle_seconds(self) -> float:
        """Seconds since this model was last used."""
        return time.monotonic() - self.last_used

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "tier": self.tier,
            "idle_seconds": round(self.idle_seconds, 1),
            "request_count": self.request_count,
            "estimated_memory_mb": self.estimated_memory_mb,
        }


# ── Active Session ────────────────────────────────────────────────────────────


@dataclass
class ActiveSession:
    """Tracks an active user session."""

    session_id: str
    user_id: str = "default"
    started_at: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    command: str = ""                    # Current command being executed
    priority: str = "P0_INTERACTIVE"     # Priority level

    def touch(self, command: str = "") -> None:
        self.last_activity = time.monotonic()
        if command:
            self.command = command

    @property
    def duration_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "duration_seconds": round(self.duration_seconds, 1),
            "command": self.command,
            "priority": self.priority,
        }


# ── Daemon Status ─────────────────────────────────────────────────────────────


class DaemonStatus(str, enum.Enum):
    """Lifecycle state of the daemon process."""

    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"


# ── Change Listener ───────────────────────────────────────────────────────────

# Type alias for state change callbacks.
# Called with (field_name, old_value, new_value).
StateChangeCallback = Callable[[str, Any, Any], Coroutine[Any, Any, None]]


# ── Daemon State ──────────────────────────────────────────────────────────────


class DaemonState:
    """Single source of truth for the running daemon.

    All reads/writes go through async methods protected by a lock.
    Change callbacks fire AFTER the lock is released to prevent deadlocks.

    Usage:
        state = DaemonState()
        state.on_change(my_callback)

        await state.set_status(DaemonStatus.RUNNING)
        await state.register_model(LoadedModel(...))
        snap = await state.snapshot()
    """

    def __init__(self, topology_name: str = "auto") -> None:
        self._lock = asyncio.Lock()
        self._callbacks: list[StateChangeCallback] = []

        # ── Lifecycle ──
        self.status: DaemonStatus = DaemonStatus.STOPPED
        self.pid: int = os.getpid()
        self.started_at: float = 0.0       # Monotonic

        # ── Topology ──
        self.topology_name: str = topology_name

        # ── Model registry ──
        self._models: dict[str, LoadedModel] = {}  # model_id → LoadedModel

        # ── Active sessions ──
        self._sessions: dict[str, ActiveSession] = {}  # session_id → ActiveSession

        # ── Connection health ──
        self.redis_health = ConnectionHealth()
        self.postgres_health = ConnectionHealth()
        self.bodega_health = ConnectionHealth()

    # ── Change observation ────────────────────────────────────────────────

    def on_change(self, callback: StateChangeCallback) -> None:
        """Register a callback for state changes."""
        self._callbacks.append(callback)

    async def _notify(self, field_name: str, old_value: Any, new_value: Any) -> None:
        """Fire all registered callbacks. Errors are logged, never raised."""
        for cb in self._callbacks:
            try:
                await cb(field_name, old_value, new_value)
            except Exception as exc:
                logger.warning(
                    "state_change_callback_error",
                    field=field_name,
                    error=str(exc),
                )

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def set_status(self, status: DaemonStatus) -> None:
        async with self._lock:
            old = self.status
            self.status = status
            if status == DaemonStatus.RUNNING and self.started_at == 0.0:
                self.started_at = time.monotonic()
        await self._notify("status", old.value, status.value)

    @property
    def uptime_seconds(self) -> float:
        if self.started_at == 0.0:
            return 0.0
        return time.monotonic() - self.started_at

    # ── Model registry ────────────────────────────────────────────────────

    async def register_model(self, model: LoadedModel) -> None:
        """Register a loaded model."""
        async with self._lock:
            self._models[model.model_id] = model
        await self._notify("model_loaded", None, model.model_id)

    async def unregister_model(self, model_id: str) -> LoadedModel | None:
        """Remove a model from the registry. Returns the removed model."""
        async with self._lock:
            model = self._models.pop(model_id, None)
        if model:
            await self._notify("model_unloaded", model.model_id, None)
        return model

    async def touch_model(self, model_id: str) -> None:
        """Record a model usage (updates last_used, increments count)."""
        async with self._lock:
            model = self._models.get(model_id)
            if model:
                model.touch()

    async def get_model(self, model_id: str) -> LoadedModel | None:
        """Get model info by ID."""
        async with self._lock:
            return self._models.get(model_id)

    async def get_all_models(self) -> list[LoadedModel]:
        """Get all loaded models."""
        async with self._lock:
            return list(self._models.values())

    async def get_idle_models(self, idle_threshold_sec: float) -> list[LoadedModel]:
        """Get models that have been idle longer than threshold."""
        async with self._lock:
            return [
                m for m in self._models.values()
                if m.idle_seconds >= idle_threshold_sec
            ]

    # ── Session tracking ──────────────────────────────────────────────────

    async def register_session(self, session: ActiveSession) -> None:
        """Track a new active session."""
        async with self._lock:
            self._sessions[session.session_id] = session
        await self._notify("session_started", None, session.session_id)

    async def unregister_session(self, session_id: str) -> ActiveSession | None:
        """Remove a completed session."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            await self._notify("session_ended", session.session_id, None)
        return session

    async def touch_session(self, session_id: str, command: str = "") -> None:
        """Record session activity."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch(command)

    async def get_active_sessions(self) -> list[ActiveSession]:
        """Get all active sessions."""
        async with self._lock:
            return list(self._sessions.values())

    # ── Connection health ─────────────────────────────────────────────────

    async def update_health(
        self,
        service: str,
        healthy: bool,
        latency_ms: float = 0.0,
        error: str = "",
    ) -> None:
        """Update health status for a service.

        Args:
            service:  "redis", "postgres", or "bodega"
            healthy:  True if check passed
            latency_ms: Response time
            error:    Error message if unhealthy
        """
        health_map = {
            "redis": self.redis_health,
            "postgres": self.postgres_health,
            "bodega": self.bodega_health,
        }
        health = health_map.get(service)
        if health is None:
            return

        async with self._lock:
            old_status = health.status.value
            if healthy:
                health.mark_healthy(latency_ms)
            else:
                health.mark_failed(error)

        if old_status != health.status.value:
            await self._notify(f"{service}_health", old_status, health.status.value)

    # ── Snapshot ──────────────────────────────────────────────────────────

    async def snapshot(self) -> dict[str, Any]:
        """Full serializable state snapshot for monitoring and status."""
        async with self._lock:
            return {
                "status": self.status.value,
                "pid": self.pid,
                "uptime_seconds": round(self.uptime_seconds, 1),
                "topology": self.topology_name,
                "models": {
                    mid: m.to_dict() for mid, m in self._models.items()
                },
                "sessions": {
                    sid: s.to_dict() for sid, s in self._sessions.items()
                },
                "connections": {
                    "redis": self.redis_health.to_dict(),
                    "postgres": self.postgres_health.to_dict(),
                    "bodega": self.bodega_health.to_dict(),
                },
            }
