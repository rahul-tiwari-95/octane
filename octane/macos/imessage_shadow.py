"""octane/macos/imessage_shadow.py — iMessage monitoring shadow.

Polls ~/Library/Messages/chat.db for new incoming messages.
Routes them through ChatEngine and replies via AppleScript.

Requires Full Disk Access for chat.db reading.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("octane.macos.imessage")

CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"

# Apple's Core Data reference date: 2001-01-01 00:00:00 UTC
_APPLE_EPOCH_OFFSET = 978307200


def _apple_ts_to_datetime(ts: int) -> datetime:
    """Convert Apple Core Data nanosecond timestamp to datetime."""
    unix_ts = ts / 1_000_000_000 + _APPLE_EPOCH_OFFSET
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc)


def _extract_body_text(attributed_body: bytes | None) -> str:
    """Extract plain text from attributedBody typedstream blob.

    Modern macOS stores message text in attributedBody as an Apple
    typedstream (NSAttributedString).  The plain text sits after the
    NSString marker + fixed overhead bytes, preceded by a length byte.
    """
    if not attributed_body:
        return ""
    try:
        blob = bytes(attributed_body)
        # Find the NSString marker
        marker = b"NSString"
        idx = blob.find(marker)
        if idx == -1:
            return ""
        # Skip past marker + fixed overhead: \x01\x94\x84\x01\x2b
        pos = idx + len(marker) + 5  # 8 + 5 = 13 bytes past marker start
        if pos >= len(blob):
            return ""
        # Read length byte (single byte for messages up to 255 chars)
        length = blob[pos]
        pos += 1
        if pos + length > len(blob):
            return ""
        text = blob[pos:pos + length].decode("utf-8", errors="replace").strip()
        return text
    except Exception:
        return ""


class IMessageShadow:
    """Background monitor for incoming iMessages.

    Polls chat.db for new messages from approved contacts,
    routes them through ChatEngine, and replies via AppleScript.
    """

    def __init__(
        self,
        approved_contacts: list[str],
        poll_interval: float = 5.0,
        max_message_length: int = 2000,
        allow_self: bool = False,
    ):
        """
        Args:
            approved_contacts: Phone numbers/emails allowed to interact.
            poll_interval: Seconds between chat.db polls.
            max_message_length: Truncate AI responses at this length.
            allow_self: Also pick up is_from_me=1 messages (for iPhone→Mac self-messaging).
        """
        self.approved_contacts = {c.strip() for c in approved_contacts}
        self.poll_interval = poll_interval
        self.max_message_length = max_message_length
        self.allow_self = allow_self
        self._last_rowid: int = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        # Track recent replies to prevent infinite echo loops in self-mode
        self._recent_replies: set[str] = set()
        # Per-sender conversation history for multi-turn context
        self._conversation_history: dict[str, list[dict[str, str]]] = {}

    def _log(self, msg: str, *args: object) -> None:
        """Print + log for foreground visibility."""
        text = msg % args if args else msg
        print(f"[iMessage] {text}", flush=True)
        logger.info(text)

    def _get_latest_rowid(self) -> int:
        """Get the current max ROWID from chat.db."""
        try:
            conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
            cursor = conn.execute("SELECT MAX(ROWID) FROM message")
            row = cursor.fetchone()
            conn.close()
            rowid = row[0] or 0
            self._log("Latest ROWID in chat.db: %d", rowid)
            return rowid
        except (sqlite3.Error, PermissionError) as exc:
            self._log("ERROR Cannot read chat.db: %s", exc)
            return 0

    def _debug_dump_recent(self, after_rowid: int) -> None:
        """Print the last 10 messages from chat.db for diagnostics (no filters)."""
        try:
            conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
            cursor = conn.execute("""
                SELECT m.ROWID, m.text, m.is_from_me, m.handle_id,
                       h.id AS sender, m.service, m.type,
                       m.attributedBody
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.ROWID > ?
                ORDER BY m.ROWID ASC
                LIMIT 10
            """, (after_rowid,))
            rows = cursor.fetchall()
            conn.close()
            if rows:
                self._log("DEBUG DUMP — %d raw messages after ROWID %d:", len(rows), after_rowid)
                for row in rows:
                    rowid, text, is_from_me, handle_id, sender, service, msg_type, attr_body = row
                    body = text or _extract_body_text(attr_body)
                    self._log(
                        "  ROWID=%d is_from_me=%s handle_id=%s sender=%s service=%s type=%s text=%r attr_body=%s body=%r",
                        rowid, is_from_me, handle_id, sender, service, msg_type,
                        (text or "")[:80],
                        f"{len(attr_body)}bytes" if attr_body else "None",
                        body[:80],
                    )
            else:
                self._log("DEBUG DUMP — no messages at all after ROWID %d", after_rowid)
        except Exception as exc:
            self._log("DEBUG DUMP ERROR: %s", exc)

    def _poll_new_messages(self) -> list[dict]:
        """Read new incoming messages since _last_rowid."""
        try:
            conn = sqlite3.connect(f"file:{CHAT_DB}?mode=ro", uri=True)
            if self.allow_self:
                # Self-mode: pick up ALL messages (both is_from_me=0 and 1).
                # We filter duplicates/echoes in _handle_message instead.
                # Include attributedBody because modern macOS often has text=NULL
                # and stores the real text in the attributedBody blob.
                query = """
                    SELECT m.ROWID, m.text, m.date, m.is_from_me,
                           h.id AS sender, m.attributedBody
                    FROM message m
                    LEFT JOIN handle h ON m.handle_id = h.ROWID
                    WHERE m.ROWID > ?
                    ORDER BY m.ROWID ASC
                    LIMIT 20
                """
            else:
                query = """
                    SELECT m.ROWID, m.text, m.date, m.is_from_me,
                           h.id AS sender, m.attributedBody
                    FROM message m
                    LEFT JOIN handle h ON m.handle_id = h.ROWID
                    WHERE m.ROWID > ?
                      AND m.is_from_me = 0
                    ORDER BY m.ROWID ASC
                    LIMIT 20
                """
            cursor = conn.execute(query, (self._last_rowid,))
            messages = []
            for row in cursor.fetchall():
                rowid, text, date_ts, is_from_me, sender, attr_body = row
                # Extract text: prefer text column, fall back to attributedBody blob
                body = (text or "").strip() or _extract_body_text(attr_body)
                if not body:
                    # No text at all (e.g. attachment-only message) — skip
                    self._last_rowid = max(self._last_rowid, rowid)
                    continue
                self._last_rowid = max(self._last_rowid, rowid)
                messages.append({
                    "rowid": rowid,
                    "text": body,
                    "timestamp": _apple_ts_to_datetime(date_ts),
                    "sender": sender or "unknown",
                    "is_from_me": bool(is_from_me),
                })
            conn.close()
            if messages:
                self._log("POLL found %d new message(s):", len(messages))
                for m in messages:
                    self._log(
                        "  ROWID=%d is_from_me=%s sender=%s text=%r",
                        m["rowid"], m["is_from_me"], m["sender"], m["text"][:80],
                    )
            return messages
        except (sqlite3.Error, PermissionError) as exc:
            self._log("ERROR chat.db poll: %s", exc)
            return []

    async def _handle_message(self, msg: dict) -> None:
        """Process a single incoming message."""
        sender = msg["sender"]
        is_from_me = msg.get("is_from_me", False)
        text = msg["text"].strip()
        if not text:
            self._log("HANDLE skipping empty text for ROWID=%d", msg["rowid"])
            return

        # Self-mode handling
        if self.allow_self:
            # In self-chat, iMessage creates TWO rows per message:
            #   is_from_me=1 (the sent copy) — skip this
            #   is_from_me=0 (the received copy) — process this as user query
            if is_from_me:
                self._log("HANDLE self-mode: skipping is_from_me=1 (sent copy) ROWID=%d", msg["rowid"])
                return
            # Skip Octane's own replies echoing back (prevent infinite loop)
            if text in self._recent_replies:
                self._recent_replies.discard(text)
                self._log("HANDLE skipping own-reply echo: %r", text[:60])
                return
            # Reply goes back to self via the approved contact
            sender = next(iter(self.approved_contacts))
            self._log("HANDLE self-mode: using received copy (is_from_me=0), sender→%s", sender)
        elif is_from_me:
            self._log("HANDLE skipping is_from_me=1 (not self-mode)")
            return
        elif sender not in self.approved_contacts:
            self._log("HANDLE ignoring unapproved sender: %s", sender)
            return

        self._log("HANDLE processing: sender=%s text=%r", sender, text[:80])

        try:
            from octane.macos.applescript import AppleScriptBridge

            # Lazy-init the ChatEngine with a real Bodega client (same as octane chat)
            if not hasattr(self, "_engine"):
                self._log("ENGINE initializing Orchestrator + Bodega...")
                try:
                    from octane.osa.orchestrator import Orchestrator
                    from octane.osa.chat_engine import ChatEngine
                    from octane.models.synapse import SynapseEventBus

                    synapse = SynapseEventBus()
                    osa = Orchestrator(synapse)
                    self._log("ENGINE calling pre_flight (non-blocking)...")
                    await osa.pre_flight(wait_for_bodega=False)
                    self._log("ENGINE pre_flight done. bodega=%s", type(osa.bodega).__name__ if osa.bodega else "None")
                    self._engine = ChatEngine(
                        bodega=osa.bodega,
                        orchestrator=osa,
                    )
                    self._log("ENGINE ChatEngine created. bodega=%s", "OK" if self._engine._bodega else "NONE!")
                except Exception as init_exc:
                    self._log("ENGINE INIT FAILED: %s", init_exc)
                    import traceback
                    traceback.print_exc()
                    return
            else:
                self._log("ENGINE already initialized, reusing")

            # ChatEngine.respond() is an async iterator that yields chunks.
            session_id = f"imessage_{sender}"
            history = self._conversation_history.get(sender, [])
            self._log("RESPOND calling ChatEngine.respond(query=%r, session=%s, history_len=%d)",
                       text[:60], session_id, len(history))

            chunks: list[str] = []
            async for chunk in self._engine.respond(text, session_id, history):
                chunks.append(chunk)
            response = "".join(chunks)

            self._log("RESPOND got %d chunks, total %d chars", len(chunks), len(response))
            self._log("RESPOND response[:200] = %r", response[:200])

            if not response.strip():
                self._log("RESPOND WARNING: empty response, not sending")
                return

            # Update conversation history (keep last 20 turns)
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": response})
            self._conversation_history[sender] = history[-20:]

            # Truncate if needed
            if len(response) > self.max_message_length:
                response = response[: self.max_message_length - 3] + "..."

            # Track this reply so we don't re-process it when it appears in chat.db
            self._recent_replies.add(response)
            if len(self._recent_replies) > 50:
                self._recent_replies = set(list(self._recent_replies)[-25:])

            bridge = AppleScriptBridge()
            self._log("SEND sending %d chars to %s via AppleScript...", len(response), sender)
            ok, status = await bridge.send_imessage(sender, response)

            if ok:
                self._log("SEND OK — replied to %s (%d chars)", sender, len(response))
            else:
                self._log("SEND FAILED — to %s: %s", sender, status)
        except Exception as exc:
            self._log("HANDLE ERROR: %s", exc)
            import traceback
            traceback.print_exc()

    async def _poll_loop(self) -> None:
        """Main polling loop."""
        self._log(
            "Shadow started — monitoring %d contacts, polling every %.0fs, self_mode=%s",
            len(self.approved_contacts),
            self.poll_interval,
            self.allow_self,
        )
        self._log("Approved contacts: %s", self.approved_contacts)

        # Seed _last_rowid to current max so we only respond to NEW messages
        self._last_rowid = self._get_latest_rowid()
        self._log("Seeded at ROWID %d — only processing messages after this", self._last_rowid)

        poll_count = 0
        while self._running:
            try:
                poll_count += 1
                saved_rowid = self._last_rowid  # save for debug dump
                messages = await asyncio.get_event_loop().run_in_executor(
                    None, self._poll_new_messages
                )
                if messages:
                    self._log("POLL #%d: %d message(s) to process", poll_count, len(messages))
                    for msg in messages:
                        await self._handle_message(msg)
                else:
                    # Debug dump on first 5 polls, then every 12th poll, to show raw DB state
                    if poll_count <= 5 or poll_count % 12 == 0:
                        self._log("POLL #%d: no filtered messages. Running debug dump...", poll_count)
                        await asyncio.get_event_loop().run_in_executor(
                            None, self._debug_dump_recent, saved_rowid
                        )
            except Exception as exc:
                self._log("POLL LOOP ERROR: %s", exc)
                import traceback
                traceback.print_exc()

            await asyncio.sleep(self.poll_interval)

    async def start(self) -> None:
        """Start the iMessage monitoring loop."""
        if self._running:
            logger.warning("iMessage shadow already running")
            return

        if not CHAT_DB.exists():
            raise FileNotFoundError(
                f"chat.db not found at {CHAT_DB}. "
                "Ensure Messages is configured and Full Disk Access is granted."
            )

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("iMessage shadow stopped")

    @property
    def is_running(self) -> bool:
        return self._running
