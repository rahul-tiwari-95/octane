"""Daemon Unix Socket Server — IPC between CLI and daemon.

Protocol: newline-delimited JSON (NDJSON).

    CLI → Daemon (request):
        {"id": "uuid", "command": "ask", "payload": {...}}

    Daemon → CLI (response):
        {"id": "uuid", "status": "ok", "data": {...}}
        {"id": "uuid", "status": "stream", "chunk": "..."}
        {"id": "uuid", "status": "done"}
        {"id": "uuid", "status": "error", "error": "..."}

The server uses asyncio.start_unix_server for zero-overhead local IPC.
Each client connection spawns a handler coroutine that reads requests,
dispatches them through the priority queue, and streams responses back.

Socket location: ~/.octane/octane.sock
    (Configurable via OCTANE_SOCKET_PATH env var)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger().bind(component="daemon_server")

# Default socket path
DEFAULT_SOCKET_PATH = Path.home() / ".octane" / "octane.sock"


def get_socket_path() -> Path:
    """Get the daemon socket path from env or default."""
    env_path = os.environ.get("OCTANE_SOCKET_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_SOCKET_PATH


# An async iterator that yields response dicts
AsyncResponseIterator = Any  # AsyncIterator[dict[str, Any]] — avoiding complex typing

# Type alias for request handlers
# Handler receives (command, payload) and returns a response (dict or async iterator)
RequestHandler = Callable[
    [str, dict[str, Any]],
    Coroutine[Any, Any, Any],
]


class DaemonServer:
    """Unix socket server for the Octane daemon.

    The server accepts connections, reads NDJSON requests, dispatches them
    to the appropriate handler, and streams NDJSON responses back.

    Args:
        socket_path:  Path to the Unix socket file.
        handler:      Async callable that processes requests.
                      Signature: (command, payload) → AsyncIterator[dict]
    """

    def __init__(
        self,
        socket_path: Path | None = None,
        handler: RequestHandler | None = None,
    ) -> None:
        self.socket_path = socket_path or get_socket_path()
        self._handler = handler
        self._server: asyncio.Server | None = None
        self._active_connections: int = 0
        self._total_requests: int = 0

    def set_handler(self, handler: RequestHandler) -> None:
        """Set the request handler (can be set after construction)."""
        self._handler = handler

    async def start(self) -> None:
        """Start listening on the Unix socket.

        Creates the socket directory if needed.
        Removes stale socket file if it exists.
        """
        # Ensure directory exists
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Clean up stale socket
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
                logger.info("stale_socket_removed", path=str(self.socket_path))
            except OSError as exc:
                logger.error("socket_cleanup_failed", error=str(exc))
                raise

        self._server = await asyncio.start_unix_server(
            self._handle_connection,
            path=str(self.socket_path),
        )

        # Set permissions — owner-only access
        os.chmod(self.socket_path, 0o600)

        logger.info("daemon_server_started", socket=str(self.socket_path))

    async def stop(self) -> None:
        """Stop the server and clean up the socket file."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        # Remove socket file
        try:
            if self.socket_path.exists():
                self.socket_path.unlink()
        except OSError:
            pass

        logger.info("daemon_server_stopped")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection.

        Reads NDJSON requests line by line, dispatches each to the handler,
        and writes NDJSON responses back.
        """
        self._active_connections += 1
        peer = "unix_client"

        try:
            while True:
                # Read one line (one JSON request)
                line = await reader.readline()
                if not line:
                    break  # Client disconnected

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    request = json.loads(line_str)
                except json.JSONDecodeError as exc:
                    await self._write_response(writer, {
                        "id": "unknown",
                        "status": "error",
                        "error": f"Invalid JSON: {exc}",
                    })
                    continue

                request_id = request.get("id", str(uuid.uuid4()))
                command = request.get("command", "")
                payload = request.get("payload", {})

                self._total_requests += 1

                logger.debug(
                    "request_received",
                    request_id=request_id,
                    command=command,
                )

                if self._handler is None:
                    await self._write_response(writer, {
                        "id": request_id,
                        "status": "error",
                        "error": "No handler registered",
                    })
                    continue

                # Dispatch to handler
                try:
                    response_iter = await self._handler(command, payload)

                    if hasattr(response_iter, "__aiter__"):
                        # Streaming response
                        async for chunk in response_iter:
                            chunk["id"] = request_id
                            await self._write_response(writer, chunk)
                    elif isinstance(response_iter, dict):
                        # Single response
                        response_iter["id"] = request_id
                        await self._write_response(writer, response_iter)
                    else:
                        await self._write_response(writer, {
                            "id": request_id,
                            "status": "ok",
                            "data": response_iter,
                        })

                except Exception as exc:
                    logger.error(
                        "handler_error",
                        request_id=request_id,
                        command=command,
                        error=str(exc),
                    )
                    await self._write_response(writer, {
                        "id": request_id,
                        "status": "error",
                        "error": str(exc),
                    })

        except asyncio.CancelledError:
            pass
        except ConnectionResetError:
            logger.debug("client_disconnected", peer=peer)
        except Exception as exc:
            logger.error("connection_error", error=str(exc))
        finally:
            self._active_connections -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    @staticmethod
    async def _write_response(
        writer: asyncio.StreamWriter,
        response: dict[str, Any],
    ) -> None:
        """Write a single NDJSON response line."""
        line = json.dumps(response, default=str) + "\n"
        writer.write(line.encode("utf-8"))
        await writer.drain()

    @property
    def active_connections(self) -> int:
        return self._active_connections

    def snapshot(self) -> dict[str, Any]:
        return {
            "socket_path": str(self.socket_path),
            "active_connections": self._active_connections,
            "total_requests": self._total_requests,
            "running": self._server is not None,
        }
