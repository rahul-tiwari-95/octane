"""Root conftest — shared pytest markers and global settings.

Markers
-------
unit        fast, no I/O, pure logic
integration requires Redis and/or Postgres (set OCTANE_TEST_INTEGRATION=1)
e2e         requires live Bodega + all infra (set OCTANE_TEST_E2E=1)
slow        expected to take > 5 seconds
"""

from __future__ import annotations

import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no I/O tests")
    config.addinivalue_line("markers", "integration: requires Redis / Postgres")
    config.addinivalue_line("markers", "e2e: requires live Bodega and all infra")
    config.addinivalue_line("markers", "slow: test is expected to take > 5 s")


# ── Skip guards ───────────────────────────────────────────────────────────────

requires_integration = pytest.mark.skipif(
    not os.getenv("OCTANE_TEST_INTEGRATION"),
    reason="Set OCTANE_TEST_INTEGRATION=1 to run integration tests",
)

requires_e2e = pytest.mark.skipif(
    not os.getenv("OCTANE_TEST_E2E"),
    reason="Set OCTANE_TEST_E2E=1 to run end-to-end tests",
)

requires_bodega = pytest.mark.skipif(
    not os.getenv("OCTANE_TEST_E2E"),
    reason="Requires live Bodega inference server (set OCTANE_TEST_E2E=1)",
)
