"""octane/macos/permissions.py — macOS permissions checker.

Checks for Automation permission (Messages, Notes, Calendar)
and Full Disk Access (for reading chat.db).
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger("octane.macos.permissions")

IS_MACOS = platform.system() == "Darwin"


async def check_automation_permission(app_name: str = "Messages") -> bool:
    """Check if Terminal/Python has Automation permission for an app.

    Tries a minimal AppleScript call. If it fails with -1743
    (not authorized), the permission is missing.
    """
    if not IS_MACOS:
        return False

    script = f'tell application "{app_name}" to return name'
    try:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        if proc.returncode == 0:
            return True
        err = stderr.decode()
        if "-1743" in err or "not allowed" in err.lower():
            return False
        # Other errors (app not found, etc.)
        return False
    except (asyncio.TimeoutError, Exception):
        return False


def check_full_disk_access() -> bool:
    """Check if the process has Full Disk Access.

    Tests by reading ~/Library/Messages/chat.db (requires FDA).
    """
    if not IS_MACOS:
        return False

    chat_db = Path.home() / "Library" / "Messages" / "chat.db"
    try:
        return chat_db.exists() and os.access(chat_db, os.R_OK)
    except (PermissionError, OSError):
        return False


async def check_all_permissions() -> dict[str, bool]:
    """Check all required macOS permissions.

    Returns dict with permission name → granted status.
    """
    results: dict[str, bool] = {
        "is_macos": IS_MACOS,
        "full_disk_access": False,
        "messages_automation": False,
        "notes_automation": False,
        "calendar_automation": False,
    }

    if not IS_MACOS:
        return results

    results["full_disk_access"] = check_full_disk_access()

    # Check automation permissions in parallel
    msg_perm, notes_perm, cal_perm = await asyncio.gather(
        check_automation_permission("Messages"),
        check_automation_permission("Notes"),
        check_automation_permission("Calendar"),
    )
    results["messages_automation"] = msg_perm
    results["notes_automation"] = notes_perm
    results["calendar_automation"] = cal_perm

    return results


def permission_guidance() -> str:
    """Return human-readable guidance for granting permissions."""
    return (
        "macOS Permissions Required:\n\n"
        "1. Automation (Messages, Notes, Calendar):\n"
        "   System Settings → Privacy & Security → Automation\n"
        "   Grant access to Terminal (or your IDE) for:\n"
        "     • Messages\n"
        "     • Notes\n"
        "     • Calendar\n\n"
        "2. Full Disk Access (for iMessage monitoring):\n"
        "   System Settings → Privacy & Security → Full Disk Access\n"
        "   Add Terminal (or your IDE/Python binary)\n\n"
        "After granting permissions, restart your terminal."
    )
