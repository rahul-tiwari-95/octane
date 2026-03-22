"""Tests for octane.security.airgap — AirgapManager.

All tests use temporary directories to avoid touching ~/.octane.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def _patch_airgap_file(tmp_path: Path):
    """Redirect _AIRGAP_FILE to a temp location."""
    from octane.security import airgap as airgap_mod
    return patch.object(airgap_mod, "_AIRGAP_FILE", tmp_path / ".airgap")


# ── is_airgap_active ──────────────────────────────────────────────────────────

class TestIsAirgapActive:
    def test_inactive_when_no_file(self, tmp_path):
        with _patch_airgap_file(tmp_path):
            from octane.security.airgap import is_airgap_active
            assert not is_airgap_active()

    def test_active_when_file_exists(self, tmp_path):
        flag = tmp_path / ".airgap"
        flag.write_text("{}")
        with _patch_airgap_file(tmp_path):
            from octane.security.airgap import is_airgap_active
            assert is_airgap_active()


# ── AirgapManager ─────────────────────────────────────────────────────────────

class TestAirgapManager:
    def _mgr(self, tmp_path):
        from octane.security.airgap import AirgapManager
        from octane.security import airgap as mod
        mod._OCTANE_DIR = tmp_path
        mod._AIRGAP_FILE = tmp_path / ".airgap"
        return AirgapManager()

    def test_enable_creates_flag_file(self, tmp_path):
        mgr = self._mgr(tmp_path)
        meta = mgr.enable(reason="testing")
        flag = tmp_path / ".airgap"
        assert flag.exists()
        assert meta["active"] is True
        on_disk = json.loads(flag.read_text())
        assert on_disk["reason"] == "testing"

    def test_flag_file_permissions(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.enable()
        flag = tmp_path / ".airgap"
        assert oct(flag.stat().st_mode)[-3:] == "600"

    def test_status_inactive(self, tmp_path):
        mgr = self._mgr(tmp_path)
        s = mgr.status()
        assert not s.get("active")

    def test_status_active_with_metadata(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.enable(reason="sensitive review")
        s = mgr.status()
        assert s["active"] is True
        assert s["reason"] == "sensitive review"
        assert "enabled_at" in s

    def test_disable_removes_flag(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.enable()
        assert (tmp_path / ".airgap").exists()
        result = mgr.disable()
        assert not (tmp_path / ".airgap").exists()
        assert not result["active"]

    def test_disable_when_not_active(self, tmp_path):
        mgr = self._mgr(tmp_path)
        result = mgr.disable()
        assert not result["active"]
        assert "not active" in result["message"].lower()

    def test_enable_then_disable_cycle(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.enable(reason="cycle test")
        assert mgr.status()["active"]
        mgr.disable()
        assert not mgr.status().get("active")

    def test_guard_raises_when_active(self, tmp_path):
        from octane.security.airgap import AirgapBlockedError
        mgr = self._mgr(tmp_path)
        mgr.enable()
        with pytest.raises(AirgapBlockedError, match="Airgap mode is ON"):
            mgr.guard("Bodega Intel search")

    def test_guard_passes_when_inactive(self, tmp_path):
        mgr = self._mgr(tmp_path)
        # Should not raise
        mgr.guard("any operation")

    def test_status_with_corrupted_flag_file(self, tmp_path):
        mgr = self._mgr(tmp_path)
        flag = tmp_path / ".airgap"
        flag.write_text("not-json{corruption")
        s = mgr.status()
        # Flag file exists → still reports active
        assert s["active"] is True

    def test_multiple_enables_idempotent(self, tmp_path):
        mgr = self._mgr(tmp_path)
        mgr.enable(reason="first")
        mgr.enable(reason="second")  # overwrite
        s = mgr.status()
        assert s["active"] is True
        assert s["reason"] == "second"


# ── Provenance integration ────────────────────────────────────────────────────

class TestBuildProvenance:
    def test_basic_structure(self, tmp_path):
        from octane.security import airgap as mod
        mod._AIRGAP_FILE = tmp_path / ".airgap"
        from octane.security.provenance import build_provenance

        prov = build_provenance(
            source="bodega_intel_api",
            command="octane investigate",
            trace_id="abc123",
            agent="web",
        )
        assert prov["source"] == "bodega_intel_api"
        assert prov["command"] == "octane investigate"
        assert prov["trace_id"] == "abc123"
        assert prov["agent"] == "web"
        assert "timestamp" in prov
        assert prov["airgap"] is False  # no flag file

    def test_airgap_flag_in_provenance(self, tmp_path):
        from octane.security import airgap as mod
        flag = tmp_path / ".airgap"
        flag.write_text("{}")
        mod._AIRGAP_FILE = flag
        from octane.security.provenance import build_provenance

        prov = build_provenance(source="web_scrape", command="octane ask --deep")
        assert prov["airgap"] is True

    def test_extras_included(self, tmp_path):
        from octane.security import airgap as mod
        mod._AIRGAP_FILE = tmp_path / ".airgap"
        from octane.security.provenance import build_provenance

        prov = build_provenance(
            source="test",
            command="test",
            dimension="market_cap",
            round=2,
        )
        assert prov["dimension"] == "market_cap"
        assert prov["round"] == 2

    def test_provenance_record_roundtrip(self, tmp_path):
        from octane.security import airgap as mod
        mod._AIRGAP_FILE = tmp_path / ".airgap"
        from octane.security.provenance import build_provenance, ProvenanceRecord

        prov = build_provenance(
            source="bodega_intel",
            command="octane investigate",
            trace_id="xyz",
            model="qwen-8b",
            agent="web",
            depth=2,
        )
        record = ProvenanceRecord.from_dict(prov)
        assert record.source == "bodega_intel"
        assert record.depth == 2
        assert record.model == "qwen-8b"
        # format() should not raise
        formatted = record.format()
        assert "bodega_intel" in formatted
        assert "qwen-8b" in formatted

    def test_optional_fields_excluded_when_empty(self, tmp_path):
        from octane.security import airgap as mod
        mod._AIRGAP_FILE = tmp_path / ".airgap"
        from octane.security.provenance import build_provenance

        prov = build_provenance(source="s", command="c")
        assert "model" not in prov
        assert "trace_id" not in prov
        assert "agent" not in prov
