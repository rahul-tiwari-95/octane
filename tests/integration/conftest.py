"""Integration-test conftest — skip guards and real-infra fixtures.

Integration tests require:
    OCTANE_TEST_INTEGRATION=1   (set in shell before running)
    Redis on localhost:6379
    Postgres on localhost:5432 with octane DB

Run with:
    OCTANE_TEST_INTEGRATION=1 sandbox/oct_env/bin/pytest tests/integration/ -v
"""

from __future__ import annotations

import os
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("OCTANE_TEST_INTEGRATION"),
    reason="Set OCTANE_TEST_INTEGRATION=1 to run integration tests",
)
