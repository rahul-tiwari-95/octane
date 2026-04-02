"""Session 41 tests — macOS integration, portfolio charts, research analytics."""
from __future__ import annotations

import asyncio
import json
import platform
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ────────────────────────────────────────────────────────────────
#  Part A: macOS native integration
# ────────────────────────────────────────────────────────────────


class TestAppleScriptBridge:
    """Unit tests for octane.macos.applescript."""

    def test_escape_handles_quotes_and_backslashes(self):
        from octane.macos.applescript import AppleScriptBridge

        assert AppleScriptBridge._escape('say "hi"') == 'say \\"hi\\"'
        assert AppleScriptBridge._escape("back\\slash") == "back\\\\slash"
        assert AppleScriptBridge._escape("plain") == "plain"

    @pytest.mark.asyncio
    async def test_send_imessage_calls_osascript(self):
        from octane.macos.applescript import AppleScriptBridge

        bridge = AppleScriptBridge()
        with patch.object(bridge, "_run_osascript", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (True, "")
            ok, msg = await bridge.send_imessage("+15551234567", "hello")
            mock_run.assert_called_once()
            script = mock_run.call_args[0][0]
            assert "Messages" in script
            assert "hello" in script
            assert ok is True

    @pytest.mark.asyncio
    async def test_send_imessage_returns_failure_on_error(self):
        from octane.macos.applescript import AppleScriptBridge

        bridge = AppleScriptBridge()
        with patch.object(bridge, "_run_osascript", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (False, "error -1743")
            ok, msg = await bridge.send_imessage("+15551234567", "hello")
            assert ok is False
            assert "error" in msg.lower() or "-1743" in msg

    @pytest.mark.asyncio
    async def test_create_note_calls_osascript(self):
        from octane.macos.applescript import AppleScriptBridge

        bridge = AppleScriptBridge()
        with patch.object(bridge, "_run_osascript", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (True, "")
            ok, msg = await bridge.create_note("Test Note", "Body text")
            mock_run.assert_called_once()
            script = mock_run.call_args[0][0]
            assert "Notes" in script
            assert "Test Note" in script

    @pytest.mark.asyncio
    async def test_read_calendar_returns_list(self):
        from octane.macos.applescript import AppleScriptBridge

        bridge = AppleScriptBridge()
        with patch.object(bridge, "_run_osascript", new_callable=AsyncMock) as mock_run:
            # Lines from AppleScript start with ||| delimiter — 6 parts when split
            mock_run.return_value = (True, "|||Meeting|||2025-01-01 10:00|||2025-01-01 11:00|||Office|||Work\n")
            events = await bridge.read_calendar(24)
            assert isinstance(events, list)
            assert len(events) == 1
            assert events[0]["summary"] == "Meeting"

    @pytest.mark.asyncio
    async def test_read_calendar_empty(self):
        from octane.macos.applescript import AppleScriptBridge

        bridge = AppleScriptBridge()
        with patch.object(bridge, "_run_osascript", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = (True, "")
            events = await bridge.read_calendar(24)
            assert events == []

    @pytest.mark.asyncio
    async def test_get_hostname_calls_scutil(self):
        from octane.macos.applescript import AppleScriptBridge

        bridge = AppleScriptBridge()
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"MyMac\n", b""))
            proc.returncode = 0
            mock_exec.return_value = proc
            name = await bridge.get_hostname()
            assert name == "MyMac"


class TestPermissions:
    """Unit tests for octane.macos.permissions."""

    @pytest.mark.asyncio
    async def test_check_automation_permission_success(self):
        from octane.macos.permissions import check_automation_permission

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            mock_exec.return_value = proc
            result = await check_automation_permission("Messages")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_automation_permission_denied(self):
        from octane.macos.permissions import check_automation_permission

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b"error -1743"))
            proc.returncode = 1
            mock_exec.return_value = proc
            result = await check_automation_permission("Messages")
            assert result is False

    def test_check_full_disk_access(self):
        from octane.macos.permissions import check_full_disk_access

        with patch("os.access", return_value=True):
            with patch("octane.macos.permissions.IS_MACOS", True):
                assert check_full_disk_access() is True

    def test_check_full_disk_access_denied(self):
        from octane.macos.permissions import check_full_disk_access

        with patch("os.access", return_value=False):
            with patch("octane.macos.permissions.IS_MACOS", True):
                assert check_full_disk_access() is False

    @pytest.mark.asyncio
    async def test_check_all_permissions_returns_dict(self):
        from octane.macos.permissions import check_all_permissions

        with patch("octane.macos.permissions.IS_MACOS", True), \
             patch("octane.macos.permissions.check_automation_permission", new_callable=AsyncMock, return_value=True), \
             patch("octane.macos.permissions.check_full_disk_access", return_value=True):
            result = await check_all_permissions()
            assert isinstance(result, dict)
            assert "is_macos" in result
            assert "full_disk_access" in result
            assert "messages_automation" in result
            assert result["is_macos"] is True

    def test_permission_guidance_returns_string(self):
        from octane.macos.permissions import permission_guidance

        text = permission_guidance()
        assert isinstance(text, str)
        assert "System" in text or "Privacy" in text or "Terminal" in text


class TestIMessageShadow:
    """Unit tests for octane.macos.imessage_shadow."""

    def test_apple_ts_to_datetime(self):
        from octane.macos.imessage_shadow import _apple_ts_to_datetime

        # Apple epoch: 2001-01-01
        ts = 0  # should be 2001-01-01T00:00:00
        dt = _apple_ts_to_datetime(ts)
        assert dt.year == 2001
        assert dt.month == 1
        assert dt.day == 1

    def test_init_stores_contacts_as_set(self):
        from octane.macos.imessage_shadow import IMessageShadow

        shadow = IMessageShadow(approved_contacts=["+15551234567", "+15559876543"])
        assert isinstance(shadow.approved_contacts, set)
        assert "+15551234567" in shadow.approved_contacts
        assert len(shadow.approved_contacts) == 2

    def test_is_running_false_by_default(self):
        from octane.macos.imessage_shadow import IMessageShadow

        shadow = IMessageShadow(approved_contacts=["+15550001111"])
        assert shadow.is_running is False

    @pytest.mark.asyncio
    async def test_start_raises_if_no_chat_db(self):
        from octane.macos.imessage_shadow import IMessageShadow

        shadow = IMessageShadow(approved_contacts=["+15550001111"])
        with patch("octane.macos.imessage_shadow.CHAT_DB", Path("/nonexistent/chat.db")):
            with pytest.raises(FileNotFoundError):
                await shadow.start()

    @pytest.mark.asyncio
    async def test_handle_message_filters_unapproved(self):
        from octane.macos.imessage_shadow import IMessageShadow

        shadow = IMessageShadow(approved_contacts=["+15550001111"])
        msg = {"sender": "+15559999999", "text": "hello", "rowid": 1}
        # Should silently skip — no error
        await shadow._handle_message(msg)

    def test_poll_new_messages_returns_list(self):
        from octane.macos.imessage_shadow import IMessageShadow

        shadow = IMessageShadow(approved_contacts=["+15550001111"])
        shadow._last_rowid = 999999999
        with patch("sqlite3.connect") as mock_conn:
            cursor = MagicMock()
            cursor.fetchall.return_value = []
            mock_conn.return_value.__enter__ = MagicMock(return_value=mock_conn.return_value)
            mock_conn.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.return_value.execute.return_value = cursor
            result = shadow._poll_new_messages()
            assert isinstance(result, list)


class TestMacOSCLI:
    """Verify macos CLI registration and command presence."""

    def test_macos_app_registered(self):
        from octane.cli import macos
        assert hasattr(macos, "macos_app")

    def test_imessage_subapp_registered(self):
        from octane.cli import macos
        assert hasattr(macos, "imessage_app")

    def test_macos_in_main_cli(self):
        """macos should appear when register_all is called."""
        import typer
        from octane.cli import register_all

        test_app = typer.Typer()
        register_all(test_app)
        group_names = [
            g.typer_instance.info.name or g.name
            for g in test_app.registered_groups
        ]
        assert "macos" in group_names


# ────────────────────────────────────────────────────────────────
#  Part A: Config
# ────────────────────────────────────────────────────────────────


class TestLANConfig:
    """Verify LAN access settings exist in OctaneSettings."""

    def test_lan_access_default_false(self):
        from octane.config import OctaneSettings

        s = OctaneSettings()
        assert s.lan_access is False

    def test_lan_token_default_empty(self):
        from octane.config import OctaneSettings

        s = OctaneSettings()
        assert s.lan_token == ""


# ────────────────────────────────────────────────────────────────
#  Part B: Portfolio + Research API endpoints
# ────────────────────────────────────────────────────────────────


class TestPortfolioAPI:
    """Test portfolio API routes via FastAPI TestClient."""

    @pytest.fixture()
    def client(self):
        from octane.ui.app import create_app
        from starlette.testclient import TestClient

        return TestClient(create_app())

    def test_positions_endpoint_exists(self, client):
        with patch("octane.ui.routes.portfolio_api._get_positions", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/portfolio/positions")
            assert resp.status_code == 200
            data = resp.json()
            assert "positions" in data
            assert "count" in data

    def test_allocation_endpoint_exists(self, client):
        with patch("octane.ui.routes.portfolio_api._get_positions", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/portfolio/allocation")
            assert resp.status_code == 200
            data = resp.json()
            assert "allocations" in data

    def test_sectors_endpoint_exists(self, client):
        with patch("octane.ui.routes.portfolio_api._get_positions", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/portfolio/sectors")
            assert resp.status_code == 200

    def test_net_worth_endpoint_exists(self, client):
        with patch("octane.portfolio.store.NetWorthStore", autospec=True) as MockStore:
            instance = MockStore.return_value
            instance.list_snapshots = AsyncMock(return_value=[])
            with patch("octane.tools.pg_client.PgClient") as MockPg:
                MockPg.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                MockPg.return_value.__aexit__ = AsyncMock(return_value=False)
                resp = client.get("/api/portfolio/net-worth")
                assert resp.status_code == 200

    def test_dividends_endpoint_exists(self, client):
        with patch("octane.portfolio.store.DividendStore", autospec=True) as MockStore:
            instance = MockStore.return_value
            instance.list_dividends = AsyncMock(return_value=[])
            with patch("octane.tools.pg_client.PgClient") as MockPg:
                MockPg.return_value.__aenter__ = AsyncMock(return_value=MagicMock())
                MockPg.return_value.__aexit__ = AsyncMock(return_value=False)
                resp = client.get("/api/portfolio/dividends")
                assert resp.status_code == 200

    def test_correlation_endpoint_exists(self, client):
        with patch("octane.ui.routes.portfolio_api._get_positions", new_callable=AsyncMock, return_value=[]):
            resp = client.get("/api/portfolio/correlation")
            assert resp.status_code == 200
            data = resp.json()
            assert "tickers" in data
            assert "matrix" in data


class TestChartsAPI:
    """Test research chart API routes."""

    @pytest.fixture()
    def client(self):
        from octane.ui.app import create_app
        from starlette.testclient import TestClient

        return TestClient(create_app())

    def test_source_distribution_endpoint(self, client):
        with patch("octane.tools.pg_client.PgClient") as MockPg:
            conn = MagicMock()
            conn.fetch = AsyncMock(return_value=[])
            MockPg.return_value.__aenter__ = AsyncMock(return_value=conn)
            MockPg.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = client.get("/api/charts/source-distribution")
            assert resp.status_code == 200
            data = resp.json()
            assert "sources" in data

    def test_research_activity_endpoint(self, client):
        with patch("octane.tools.pg_client.PgClient") as MockPg:
            conn = MagicMock()
            conn.fetch = AsyncMock(return_value=[])
            MockPg.return_value.__aenter__ = AsyncMock(return_value=conn)
            MockPg.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = client.get("/api/charts/research-activity")
            assert resp.status_code == 200

    def test_trust_scores_endpoint(self, client):
        with patch("octane.tools.pg_client.PgClient") as MockPg:
            conn = MagicMock()
            conn.fetch = AsyncMock(return_value=[])
            MockPg.return_value.__aenter__ = AsyncMock(return_value=conn)
            MockPg.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = client.get("/api/charts/trust-scores")
            assert resp.status_code == 200

    def test_extraction_stats_endpoint(self, client):
        with patch("octane.tools.pg_client.PgClient") as MockPg:
            conn = MagicMock()
            conn.fetchrow = AsyncMock(return_value=None)
            MockPg.return_value.__aenter__ = AsyncMock(return_value=conn)
            MockPg.return_value.__aexit__ = AsyncMock(return_value=False)
            resp = client.get("/api/charts/extraction-stats")
            assert resp.status_code == 200


# ────────────────────────────────────────────────────────────────
#  Part B: Frontend component existence
# ────────────────────────────────────────────────────────────────


class TestFrontendComponents:
    """Verify all new frontend files exist."""

    CHART_DIR = Path(__file__).resolve().parent.parent.parent / "octane" / "ui-frontend" / "src" / "components" / "charts"
    PAGES_DIR = Path(__file__).resolve().parent.parent.parent / "octane" / "ui-frontend" / "src" / "pages"

    @pytest.mark.parametrize("component", [
        "AllocationDonut.tsx",
        "SectorRadar.tsx",
        "CorrelationHeatmap.tsx",
        "NetWorthTimeline.tsx",
        "DividendBar.tsx",
        "HoldingsTable.tsx",
        "SourceDistribution.tsx",
        "ResearchActivity.tsx",
        "TrustScores.tsx",
    ])
    def test_chart_component_exists(self, component):
        assert (self.CHART_DIR / component).is_file(), f"Missing {component}"

    def test_portfolio_page_exists(self):
        assert (self.PAGES_DIR / "PortfolioPage.tsx").is_file()

    def test_portfolio_page_imports_all_charts(self):
        content = (self.PAGES_DIR / "PortfolioPage.tsx").read_text()
        for name in [
            "AllocationDonut", "SectorRadar", "CorrelationHeatmap",
            "NetWorthTimeline", "DividendBar", "HoldingsTable",
            "SourceDistribution", "ResearchActivity", "TrustScores",
        ]:
            assert name in content, f"PortfolioPage missing import of {name}"


class TestNavigationUpdated:
    """Verify App.tsx and Header.tsx were updated for portfolio route."""

    UI_SRC = Path(__file__).resolve().parent.parent.parent / "octane" / "ui-frontend" / "src"

    def test_app_tsx_has_portfolio_route(self):
        content = (self.UI_SRC / "App.tsx").read_text()
        assert "PortfolioPage" in content
        assert "onPortfolio" in content

    def test_header_has_portfolio_link(self):
        content = (self.UI_SRC / "components" / "Header.tsx").read_text()
        assert "/portfolio" in content
        assert "Portfolio" in content

    def test_app_css_has_portfolio_grid(self):
        content = (self.UI_SRC / "App.css").read_text()
        assert "portfolio-grid" in content
        assert "pf-row-top" in content
        assert "pf-row-research" in content


# ────────────────────────────────────────────────────────────────
#  Part A: macOS package structure
# ────────────────────────────────────────────────────────────────


class TestMacOSPackage:
    """Verify octane.macos package structure."""

    def test_package_importable(self):
        import octane.macos  # noqa: F401

    def test_applescript_importable(self):
        from octane.macos.applescript import AppleScriptBridge  # noqa: F401

    def test_permissions_importable(self):
        from octane.macos.permissions import check_all_permissions  # noqa: F401

    def test_imessage_shadow_importable(self):
        from octane.macos.imessage_shadow import IMessageShadow  # noqa: F401
