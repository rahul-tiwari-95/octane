"""AirgapManager — network kill switch for Octane.

When airgap mode is active, all outbound network calls are blocked:
  - BodegaIntelClient (web search, finance, news, entertainment, music)
  - Web agent fetcher (httpx, Playwright)
  - Any future networking in CIA/Shadows sync

The kill switch is a flag file at ~/.octane/.airgap with JSON metadata.
Checking it is a single Path.exists() call — zero overhead on the hot path.

Usage:
    from octane.security.airgap import is_airgap_active, AirgapManager

    # In any network client, guard outbound calls:
    if is_airgap_active():
        raise AirgapBlockedError("Network access blocked: airgap mode is ON.")

    # CLI management:
    mgr = AirgapManager()
    mgr.enable(reason="Reviewing sensitive financial data")
    mgr.disable()
    mgr.status()
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger().bind(component="security.airgap")

_OCTANE_DIR  = Path.home() / ".octane"
_AIRGAP_FILE = _OCTANE_DIR / ".airgap"


class AirgapBlockedError(Exception):
    """Raised when a network operation is attempted while airgap is active."""


def is_airgap_active() -> bool:
    """Fast check: is airgap mode currently active?

    Designed to be called on every outbound request — just a Path.exists().
    """
    return _AIRGAP_FILE.exists()


class AirgapManager:
    """Enable, disable, and inspect airgap mode."""

    # ── Enable / Disable ──────────────────────────────────────────────────────

    def enable(self, reason: str = "") -> dict:
        """Activate airgap mode. Returns status dict."""
        _OCTANE_DIR.mkdir(parents=True, exist_ok=True)
        meta = {
            "active": True,
            "enabled_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "enabled_by": os.environ.get("USER", "unknown"),
        }
        _AIRGAP_FILE.write_text(json.dumps(meta, indent=2))
        _AIRGAP_FILE.chmod(0o600)
        logger.warning("airgap.enabled", reason=reason)
        return meta

    def disable(self) -> dict:
        """Deactivate airgap mode. Returns status dict."""
        if not _AIRGAP_FILE.exists():
            return {"active": False, "message": "Airgap was not active."}
        _AIRGAP_FILE.unlink()
        logger.info("airgap.disabled")
        return {"active": False, "message": "Airgap mode disabled. Network access restored."}

    def status(self) -> dict:
        """Return current airgap status with metadata."""
        if not _AIRGAP_FILE.exists():
            return {"active": False}
        try:
            meta = json.loads(_AIRGAP_FILE.read_text())
            return meta
        except (json.JSONDecodeError, OSError):
            # Flag file exists but is unreadable — still active
            return {
                "active": True,
                "enabled_at": "unknown",
                "reason": "Flag file exists (metadata unreadable).",
            }

    def guard(self, operation: str = "network request") -> None:
        """Raise AirgapBlockedError if airgap is active.

        Call this at the top of any network operation.

        Example:
            airgap.guard("Bodega Intel search")
        """
        if is_airgap_active():
            raise AirgapBlockedError(
                f"Airgap mode is ON — {operation} blocked.\n"
                "Run 'octane airgap off' to restore network access."
            )
