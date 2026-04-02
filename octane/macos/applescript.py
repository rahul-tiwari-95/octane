"""octane/macos/applescript.py — AppleScript bridge for macOS.

Async wrappers around osascript for iMessage, Apple Notes, Calendar.
All methods are no-ops on non-macOS platforms.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import shlex

logger = logging.getLogger("octane.macos.applescript")

IS_MACOS = platform.system() == "Darwin"


class AppleScriptBridge:
    """Execute AppleScripts via osascript for macOS system integration."""

    # ── Low-level runner ──────────────────────────────────────────────────

    async def _run_osascript(self, script: str, timeout: float = 15.0) -> tuple[bool, str]:
        """Run an AppleScript string via osascript.

        Returns (success, output_or_error).
        """
        if not IS_MACOS:
            logger.warning("AppleScript not available on %s", platform.system())
            return False, "AppleScript requires macOS"

        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            if proc.returncode == 0:
                return True, stdout.decode().strip()
            else:
                err = stderr.decode().strip()
                logger.error("osascript failed: %s", err)
                return False, err
        except asyncio.TimeoutError:
            logger.error("osascript timed out after %.0fs", timeout)
            return False, "Timed out"
        except Exception as exc:
            logger.error("osascript error: %s", exc)
            return False, str(exc)

    @staticmethod
    def _escape(text: str) -> str:
        """Escape text for safe embedding in AppleScript strings."""
        return text.replace("\\", "\\\\").replace('"', '\\"')

    # ── iMessage ──────────────────────────────────────────────────────────

    async def send_imessage(self, to: str, message: str) -> tuple[bool, str]:
        """Send an iMessage to a phone number or email.

        Args:
            to: Phone number (+1...) or Apple ID email.
            message: Text to send.

        Returns:
            (success, status_message)
        """
        safe_msg = self._escape(message)
        safe_to = self._escape(to)
        script = (
            'tell application "Messages"\n'
            f'  send "{safe_msg}" to buddy "{safe_to}" of '
            '(1st service whose service type = iMessage)\n'
            'end tell'
        )
        ok, result = await self._run_osascript(script)
        if ok:
            logger.info("iMessage sent to %s (%d chars)", to, len(message))
        return ok, "Sent" if ok else result

    # ── Apple Notes ───────────────────────────────────────────────────────

    async def create_note(
        self, title: str, body: str, folder: str = "Octane"
    ) -> tuple[bool, str]:
        """Create a note in Apple Notes.

        Creates the folder if it doesn't exist.

        Args:
            title: Note title.
            body: Note body (plain text, will be wrapped in HTML).
            folder: Notes folder name (default: "Octane").
        """
        safe_title = self._escape(title)
        safe_body = self._escape(body).replace("\n", "<br>")
        safe_folder = self._escape(folder)

        script = (
            'tell application "Notes"\n'
            f'  set folderName to "{safe_folder}"\n'
            '  set targetFolder to missing value\n'
            '  repeat with f in folders\n'
            '    if name of f is folderName then\n'
            '      set targetFolder to f\n'
            '      exit repeat\n'
            '    end if\n'
            '  end repeat\n'
            '  if targetFolder is missing value then\n'
            '    set targetFolder to make new folder with properties {name:folderName}\n'
            '  end if\n'
            f'  make new note at targetFolder with properties '
            f'{{name:"{safe_title}", body:"<html><body>'
            f'<h1>{safe_title}</h1>{safe_body}</body></html>"}}\n'
            'end tell'
        )
        ok, result = await self._run_osascript(script)
        if ok:
            logger.info("Note created: %s (folder: %s)", title, folder)
        return ok, "Created" if ok else result

    # ── Calendar ──────────────────────────────────────────────────────────

    async def read_calendar(self, hours_ahead: int = 24) -> list[dict]:
        """Read upcoming calendar events from Apple Calendar.

        Args:
            hours_ahead: How many hours ahead to look.

        Returns:
            List of {summary, start_date, end_date, location, calendar} dicts.
        """
        script = (
            'set output to ""\n'
            'set now to current date\n'
            f'set horizon to now + ({hours_ahead} * 60 * 60)\n'
            'tell application "Calendar"\n'
            '  repeat with cal in calendars\n'
            '    set calName to name of cal\n'
            '    set evts to (every event of cal whose '
            'start date >= now and start date <= horizon)\n'
            '    repeat with e in evts\n'
            '      set output to output & "|||" & summary of e & '
            '"|||" & (start date of e as string) & '
            '"|||" & (end date of e as string) & '
            '"|||" & (location of e as string) & '
            '"|||" & calName & "\\n"\n'
            '    end repeat\n'
            '  end repeat\n'
            'end tell\n'
            'return output'
        )
        ok, raw = await self._run_osascript(script, timeout=30.0)
        if not ok:
            logger.error("Calendar read failed: %s", raw)
            return []

        events = []
        for line in raw.strip().splitlines():
            parts = line.split("|||")
            if len(parts) >= 6:
                events.append({
                    "summary": parts[1].strip(),
                    "start_date": parts[2].strip(),
                    "end_date": parts[3].strip(),
                    "location": parts[4].strip(),
                    "calendar": parts[5].strip(),
                })
        return events

    # ── macOS metadata ────────────────────────────────────────────────────

    async def get_hostname(self) -> str:
        """Get the Mac's network hostname."""
        ok, name = await self._run_osascript(
            'do shell script "scutil --get ComputerName"'
        )
        return name if ok else "unknown"
