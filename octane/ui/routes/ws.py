"""WebSocket routes — live Synapse events + PTY terminal bridge.

/ws/events             — broadcasts Synapse events in real-time
/ws/terminal           — session-based PTY bridge for xterm.js
/api/terminal/sessions — REST endpoints for terminal session management
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import pty
import struct
import subprocess
import termios
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
logger = logging.getLogger("octane.ui.ws")

# ── Live event broadcast ──────────────────────────────────────────────────────

_event_subscribers: set[WebSocket] = set()


async def broadcast_event(event: dict[str, Any]) -> None:
    """Push a Synapse event to all connected WebSocket clients."""
    dead: set[WebSocket] = set()
    msg = json.dumps(event, default=str)
    for ws in list(_event_subscribers):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _event_subscribers -= dead


@router.websocket("/ws/events")
async def ws_events(websocket: WebSocket) -> None:
    """Live Synapse event stream.

    Clients connect, receive JSON events, send nothing.
    Tails ~/.octane/traces/*.jsonl for events emitted by CLI processes.
    New trace files appearing after connection are read from byte 0.
    """
    await websocket.accept()
    _event_subscribers.add(websocket)
    logger.info("ws/events: client connected (%d total)", len(_event_subscribers))
    try:
        trace_dir = Path.home() / ".octane" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)

        # Record sizes of files that already exist — skip their old content.
        # Files appearing *after* connection are read from byte 0.
        seen_sizes: dict[str, int] = {}
        for tf in trace_dir.glob("*.jsonl"):
            seen_sizes[str(tf)] = tf.stat().st_size

        while True:
            if trace_dir.is_dir():
                for tf in trace_dir.glob("*.jsonl"):
                    key = str(tf)
                    try:
                        current_size = tf.stat().st_size
                    except OSError:
                        continue
                    prev_size = seen_sizes.get(key, 0)  # new files → read from 0
                    if current_size > prev_size:
                        with open(tf, "r") as f:
                            f.seek(prev_size)
                            new_lines = f.read()
                        for line in new_lines.strip().splitlines():
                            if line.strip():
                                try:
                                    event = json.loads(line)
                                    await websocket.send_text(
                                        json.dumps(event, default=str)
                                    )
                                except json.JSONDecodeError:
                                    continue
                    seen_sizes[key] = current_size
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        logger.info("ws/events: client disconnected")
    except Exception as e:
        logger.warning("ws/events: error: %s", e)
    finally:
        _event_subscribers.discard(websocket)
        logger.info(
            "ws/events: cleaned up (%d remaining)", len(_event_subscribers)
        )


# ── Terminal session pool ─────────────────────────────────────────────────────

_MAX_SCROLLBACK = 64 * 1024  # 64 KB scrollback buffer per session
_SESSION_TTL = 1800  # kill sessions inactive for 30 min


@dataclass
class TerminalSession:
    session_id: str
    proc: subprocess.Popen[bytes]
    master_fd: int
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    scrollback: bytearray = field(default_factory=bytearray)
    reader_task: asyncio.Task[None] | None = None
    ws: WebSocket | None = None
    title: str = "shell"


_sessions: dict[str, TerminalSession] = {}


def _create_session() -> TerminalSession:
    """Spawn a PTY shell and register a new session."""
    session_id = uuid.uuid4().hex[:12]
    master_fd, slave_fd = pty.openpty()
    shell = os.environ.get("SHELL", "/bin/zsh")
    venv_path = Path(__file__).resolve().parent.parent.parent.parent / ".venv"
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    env["COLUMNS"] = "120"
    env["LINES"] = "40"

    if venv_path.is_dir():
        env["VIRTUAL_ENV"] = str(venv_path)
        env["PATH"] = f"{venv_path / 'bin'}:{env.get('PATH', '')}"

    proc = subprocess.Popen(
        [shell, "-l"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave_fd)

    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    session = TerminalSession(
        session_id=session_id,
        proc=proc,
        master_fd=master_fd,
    )
    _sessions[session_id] = session
    session.reader_task = asyncio.create_task(_session_reader(session))
    logger.info("terminal session %s created (PID %d)", session_id, proc.pid)
    return session


async def _session_reader(session: TerminalSession) -> None:
    """Persistent PTY reader — sends to attached WS or buffers for replay."""
    try:
        while session.proc.poll() is None:
            try:
                data = os.read(session.master_fd, 4096)
                if not data:
                    break
                session.scrollback.extend(data)
                if len(session.scrollback) > _MAX_SCROLLBACK:
                    session.scrollback = session.scrollback[-_MAX_SCROLLBACK:]
                if session.ws is not None:
                    try:
                        await session.ws.send_text(
                            data.decode("utf-8", errors="replace")
                        )
                    except Exception:
                        session.ws = None
            except BlockingIOError:
                await asyncio.sleep(0.02)
            except OSError:
                break
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    logger.info("terminal session %s reader stopped", session.session_id)


def _kill_session(session_id: str) -> bool:
    """Terminate a session's PTY and remove it from the pool."""
    session = _sessions.pop(session_id, None)
    if session is None:
        return False
    if session.reader_task:
        session.reader_task.cancel()
    try:
        session.proc.terminate()
    except OSError:
        pass
    try:
        os.close(session.master_fd)
    except OSError:
        pass
    logger.info("terminal session %s killed", session_id)
    return True


# ── REST: terminal session management ─────────────────────────────────────────


@router.get("/api/terminal/sessions")
async def list_sessions():
    """List active terminal sessions."""
    # Garbage-collect dead sessions
    dead = [sid for sid, s in _sessions.items() if s.proc.poll() is not None]
    for sid in dead:
        _kill_session(sid)
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "title": s.title,
                "pid": s.proc.pid,
                "created_at": s.created_at,
                "last_activity": s.last_activity,
                "alive": s.proc.poll() is None,
                "attached": s.ws is not None,
            }
            for s in _sessions.values()
        ]
    }


@router.post("/api/terminal/sessions")
async def create_session_endpoint():
    """Create a new terminal session and return its ID."""
    session = _create_session()
    return {"session_id": session.session_id, "pid": session.proc.pid}


@router.delete("/api/terminal/sessions/{session_id}")
async def delete_session(session_id: str):
    """Kill a terminal session."""
    if _kill_session(session_id):
        return {"status": "killed"}
    return {"status": "not_found"}


# ── WebSocket: terminal PTY bridge ────────────────────────────────────────────


@router.websocket("/ws/terminal")
async def ws_terminal(websocket: WebSocket) -> None:
    """Session-based PTY bridge.

    Query params:
        session_id — reattach to an existing session (optional).
                     If absent or invalid, a new session is created.
    """
    await websocket.accept()

    session_id = websocket.query_params.get("session_id")
    session = _sessions.get(session_id) if session_id else None

    if session is None or session.proc.poll() is not None:
        session = _create_session()
        logger.info("ws/terminal: new session %s", session.session_id)
    else:
        logger.info("ws/terminal: reattaching to session %s", session.session_id)

    # Detach any previous WS from this session
    session.ws = websocket
    session.last_activity = time.time()

    try:
        # First message: session metadata
        await websocket.send_text(
            json.dumps(
                {
                    "type": "session_info",
                    "session_id": session.session_id,
                    "pid": session.proc.pid,
                }
            )
        )

        # Replay scrollback so reconnecting clients see prior output
        if session.scrollback:
            await websocket.send_text(
                session.scrollback.decode("utf-8", errors="replace")
            )

        # Receive loop — keystrokes + resize commands from xterm.js
        while True:
            message = await websocket.receive()
            session.last_activity = time.time()
            if "text" in message:
                text = message["text"]
                if (
                    text.startswith("{")
                    and '"type"' in text
                    and '"resize"' in text
                ):
                    try:
                        resize = json.loads(text)
                        cols = resize.get("cols", 120)
                        rows = resize.get("rows", 40)
                        winsize = struct.pack("HHHH", rows, cols, 0, 0)
                        fcntl.ioctl(
                            session.master_fd, termios.TIOCSWINSZ, winsize
                        )
                    except (json.JSONDecodeError, OSError):
                        pass
                else:
                    try:
                        os.write(session.master_fd, text.encode())
                    except OSError:
                        break
            elif "bytes" in message:
                try:
                    os.write(session.master_fd, message["bytes"])
                except OSError:
                    break
    except (WebSocketDisconnect, RuntimeError):
        logger.info(
            "ws/terminal: client disconnected from session %s",
            session.session_id,
        )
    except Exception as e:
        logger.error(
            "ws/terminal: error in session %s: %s",
            session.session_id,
            e,
        )
    finally:
        # Detach WS but keep the session alive for reconnect
        if session.ws is websocket:
            session.ws = None
        logger.info(
            "ws/terminal: detached from %s (session stays alive)",
            session.session_id,
        )
