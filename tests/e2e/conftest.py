"""E2E conftest — requires live Bodega and all infrastructure.

E2E tests exercise the full research pipeline end-to-end:
    - Live Bodega inference server (localhost:44468)
    - Redis + Postgres
    - Real HTTP calls to Brave Search API

Run with:
    OCTANE_TEST_E2E=1 sandbox/oct_env/bin/pytest tests/e2e/ -v
"""

from __future__ import annotations

import os
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("OCTANE_TEST_E2E"),
    reason="Set OCTANE_TEST_E2E=1 to run end-to-end tests",
)
