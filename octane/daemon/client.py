"""Daemon Client — CLI-side IPC for communicating with the daemon.

The client auto-detects whether the daemon is running by checking:
    1. Does the socket file exist?
    2. Is the PID in the PID file still alive?

If the daemon is available, CLI commands route through it (shared pools,
warm caches, priority scheduling). If not, commands fall back to direct
execution — the daemon is ALWAYS optional.

Protocol: NDJSON over Unix socket (same as server.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import structlog

from octane.daemon.server import get_socket_path

logger = structlog.get_logger().bind(component="daemon_client")

# PID file location
DEFAULT_PID_PATH = Path.home() / ".octane" / "daemon.pid"


def get_pid_path() -> Path:
    """Get the PID file path."""
    env_path = os.environ.get("OCTANE_PID_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_PID_PATH


def is_daemon_running() -> bool:
    """Check if the daemon is running.

    Checks:
        1. Socket file exists
        2. PID file exists and PID is alive

    Returns True only if both conditions are met.
    This is a synchronous check — safe to call from anywhere.
    """
    socket_path = get_socket_path()
    pid_path = get_pid_path()

    if not socket_path.exists():
        return False

    if not pid_path.exists():
        # Socket exists but no PID file — stale socket
        _cleanup_stale_socket(socket_path)
        return False

    try:
        pid = int(pid_path.read_text().strip())
    except (ValueError, OSError):
        _cleanup_stale_socket(socket_path)
        return False

    # Check if PID is alive
    if not _pid_alive(pid):
        _cleanup_stale(socket_path, pid_path)
        return False

    return True


def get_daemon_pid() -> int | None:
    """Read the daemon PID from the PID file. Returns None if not available."""
    pid_path = get_pid_path()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)  # Signal 0 = check existence, don't actually kill
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it (shouldn't happen for own daemon)
        return True


def _cleanup_stale_socket(socket_path: Path) -> None:
    """Remove a stale socket file."""
    try:
        socket_path.unlink(missing_ok=True)
        logger.info("stale_socket_cleaned", path=str(socket_path))
    except OSError:
        pass


def _cleanup_stale(socket_path: Path, pid_path: Path) -> None:
    """Remove stale socket and PID files."""
    _cleanup_stale_socket(socket_path)
    try:
        pid_path.unlink(missing_ok=True)
        logger.info("stale_pid_cleaned", path=str(pid_path))
    except OSError:
        pass


class DaemonClient:
    """Async client for communicating with the Octane daemon over Unix socket.

    Usage:
        client = DaemonClient()
        if not await client.connect():
            # Daemon not running — fall back to direct execution
            return

        # Single request/response
        result = await client.request("status", {})

        # Streaming request
        async for chunk in client.stream("ask", {"query": "..."}):
            print(chunk)

        await client.close()
    """

    def __init__(self, socket_path: Path | None = None) -> None:
        self.socket_path = socket_path or get_socket_path()
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self, timeout: float = 5.0) -> bool:
        """Connect to the daemon socket.

        Returns True if connected, False if daemon unavailable.
        Never raises — graceful degradation.
        """
        if not self.socket_path.exists():
            return False

        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self.socket_path)),
                timeout=timeout,
            )
            self._connected = True
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError, FileNotFoundError,
                OSError) as exc:
            logger.debug("daemon_connect_failed", error=str(exc))
            self._connected = False
            return False

    async def close(self) -> None:
        """Close the connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        self._connected = False

    async def request(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        """Send a request and wait for a single response.

        Args:
            command:  Command name (e.g., "status", "ask", "health").
            payload:  Command-specific data.
            timeout:  Max seconds to wait for response.

        Returns:
            Response dict with "status", "data"/"error" fields.

        Raises:
            ConnectionError: If not connected.
        """
        if not self._connected or self._writer is None or self._reader is None:
            raise ConnectionError("Not connected to daemon")

        import uuid
        request_id = str(uuid.uuid4())

        request = {
            "id": request_id,
            "command": command,
            "payload": payload or {},
        }

        # Send request
        line = json.dumps(request, default=str) + "\n"
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()

        # Read response
        try:
            response_line = await asyncio.wait_for(
                self._reader.readline(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {"id": request_id, "status": "error", "error": "Request timed out"}

        if not response_line:
            return {"id": request_id, "status": "error", "error": "Connection closed"}

        try:
            return json.loads(response_line.decode("utf-8").strip())
        except json.JSONDecodeError as exc:
            return {"id": request_id, "status": "error", "error": f"Invalid response: {exc}"}

    async def stream(
        self,
        command: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a request and stream responses.

        Yields response dicts until a "done" or "error" status is received.
        """
        if not self._connected or self._writer is None or self._reader is None:
            raise ConnectionError("Not connected to daemon")

        import uuid
        request_id = str(uuid.uuid4())

        request = {
            "id": request_id,
            "command": command,
            "payload": payload or {},
        }

        line = json.dumps(request, default=str) + "\n"
        self._writer.write(line.encode("utf-8"))
        await self._writer.drain()

        deadline = asyncio.get_event_loop().time() + timeout

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                yield {"id": request_id, "status": "error", "error": "Stream timed out"}
                break

            try:
                response_line = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=min(remaining, 30.0),
                )
            except asyncio.TimeoutError:
                continue  # Keep waiting until deadline

            if not response_line:
                yield {"id": request_id, "status": "error", "error": "Connection closed"}
                break

            try:
                response = json.loads(response_line.decode("utf-8").strip())
            except json.JSONDecodeError:
                continue

            yield response

            status = response.get("status", "")
            if status in ("done", "error"):
                break
