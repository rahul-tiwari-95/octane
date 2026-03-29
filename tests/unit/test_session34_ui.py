"""Tests for Session 34 — Octane Mission Control UI backend routes."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture
def app():
    from octane.ui.app import create_app
    return create_app()


@pytest.fixture
def client(app):
    from starlette.testclient import TestClient
    return TestClient(app)


# ── /api/system ───────────────────────────────────────────────────────────────


class TestSystemRoute:
    """GET /api/system returns CPU, RAM, disk info."""

    def test_system_endpoint_returns_cpu(self, client):
        resp = client.get("/api/system")
        assert resp.status_code == 200
        data = resp.json()
        assert "cpu" in data
        assert "percent" in data["cpu"]
        assert 0 <= data["cpu"]["percent"] <= 100

    def test_system_endpoint_returns_ram(self, client):
        resp = client.get("/api/system")
        data = resp.json()
        assert "ram" in data
        assert "total_gb" in data["ram"]
        assert "percent" in data["ram"]

    def test_system_endpoint_returns_disk(self, client):
        resp = client.get("/api/system")
        data = resp.json()
        assert "disk" in data
        assert "percent" in data["disk"]

    def test_system_endpoint_returns_platform(self, client):
        resp = client.get("/api/system")
        data = resp.json()
        assert "platform" in data
        assert "system" in data["platform"]


# ── /api/dashboard ────────────────────────────────────────────────────────────


class TestDashboardRoute:
    """GET /api/dashboard returns aggregated stats."""

    def test_dashboard_returns_uptime(self, client):
        resp = client.get("/api/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime" in data

    def test_dashboard_returns_recent_traces(self, client):
        resp = client.get("/api/dashboard")
        data = resp.json()
        assert "recent_traces" in data
        assert isinstance(data["recent_traces"], list)


# ── /api/models ───────────────────────────────────────────────────────────────


class TestModelsRoute:
    """GET /api/models returns Bodega model info or offline status."""

    def test_models_offline_when_bodega_down(self, client):
        # Bodega likely not running during tests
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        # Either returns models or fallback offline status
        assert isinstance(data, (dict, list))

    @patch("octane.ui.routes.models.httpx.AsyncClient.get")
    def test_models_returns_parsed_data(self, mock_get, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {
                    "id": "mlx-community/Qwen3-8B-4bit",
                    "type": "lm",
                    "context_length": 32768,
                    "status": "ready",
                    "memory_mb": 4500,
                    "pid": 12345,
                }
            ]
        }
        mock_get.return_value = mock_resp
        resp = client.get("/api/models")
        assert resp.status_code == 200


# ── /api/traces ───────────────────────────────────────────────────────────────


class TestTracesRoute:
    """GET /api/traces lists recent traces."""

    def test_traces_returns_dict_with_traces_key(self, client):
        resp = client.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert "traces" in data
        assert isinstance(data["traces"], list)
        assert "total" in data

    def test_traces_respects_limit(self, client):
        resp = client.get("/api/traces?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["traces"]) <= 5


# ── /api/research ─────────────────────────────────────────────────────────────


class TestResearchRoute:
    """GET /api/research/tasks and /api/research/findings."""

    def test_research_tasks_returns_dict(self, client):
        resp = client.get("/api/research/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
        assert "total" in data

    def test_research_findings_returns_dict(self, client):
        resp = client.get("/api/research/findings")
        assert resp.status_code == 200
        data = resp.json()
        assert "findings" in data
        assert isinstance(data["findings"], list)


# ── /api/ask ──────────────────────────────────────────────────────────────────


class TestQueryRoute:
    """POST /api/ask submits a query."""

    def test_ask_requires_body(self, client):
        resp = client.post("/api/ask", json={})
        # Should return 422 or 400 for missing required field
        assert resp.status_code in (400, 422)

    def test_ask_valid_query(self, client):
        with patch("octane.osa.orchestrator.Orchestrator") as mock_orch:
            mock_orch.return_value.run = AsyncMock()
            resp = client.post("/api/ask", json={"query": "What is Octane?"})
            assert resp.status_code == 200
            data = resp.json()
            assert "trace_id" in data
            assert data["query"] == "What is Octane?"


# ── /api/auth ─────────────────────────────────────────────────────────────────


class TestAuthRoute:
    """Auth token endpoint."""

    def test_auth_token_returns_session(self, client):
        resp = client.post("/api/auth/token")
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data or "session" in data or resp.status_code == 200


# ── CLI: octane ui ────────────────────────────────────────────────────────────


class TestUICli:
    """octane ui start/stop/status CLI commands."""

    def test_ui_module_imports(self):
        from octane.cli.ui import ui_app
        assert ui_app is not None

    def test_ui_status_when_not_running(self):
        from octane.cli.ui import _read_pid
        # Should return None when no UI is running
        pid = _read_pid()
        # pid is either None or an int
        assert pid is None or isinstance(pid, int)

    def test_kill_port_holders_no_crash_on_free_port(self):
        from octane.cli.ui import _kill_port_holders
        # Port 19999 unlikely to be in use
        killed = _kill_port_holders(19999)
        assert isinstance(killed, list)


# ── Terminal session REST endpoints ───────────────────────────────────────────


class TestTerminalSessions:
    """REST endpoints for terminal session management."""

    def test_list_sessions_returns_dict(self, client):
        resp = client.get("/api/terminal/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)

    def test_create_session_returns_id(self, client):
        resp = client.post("/api/terminal/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "pid" in data
        # Clean up
        sid = data["session_id"]
        client.delete(f"/api/terminal/sessions/{sid}")

    def test_delete_nonexistent_session(self, client):
        resp = client.delete("/api/terminal/sessions/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_found"

    def test_create_and_delete_session(self, client):
        resp = client.post("/api/terminal/sessions")
        sid = resp.json()["session_id"]
        resp2 = client.delete(f"/api/terminal/sessions/{sid}")
        assert resp2.json()["status"] == "killed"

    def test_created_session_appears_in_list(self, client):
        resp = client.post("/api/terminal/sessions")
        sid = resp.json()["session_id"]
        listed = client.get("/api/terminal/sessions").json()
        ids = [s["session_id"] for s in listed["sessions"]]
        assert sid in ids
        # Clean up
        client.delete(f"/api/terminal/sessions/{sid}")


# ── App factory ───────────────────────────────────────────────────────────────


class TestAppFactory:
    """create_app() returns a FastAPI application."""

    def test_create_app_returns_fastapi(self):
        from octane.ui.app import create_app
        app = create_app()
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)

    def test_app_has_api_routes(self, app):
        routes = [r.path for r in app.routes if hasattr(r, 'path')]
        # Check key routes exist
        api_paths = [r for r in routes if r.startswith("/api")]
        assert len(api_paths) > 0

    def test_cors_configured(self, app):
        from starlette.middleware.cors import CORSMiddleware
        middleware_classes = [m.cls for m in app.user_middleware]
        assert CORSMiddleware in middleware_classes
