"""Integration-test conftest — skip guards and real-infra fixtures.

Integration tests require:
    OCTANE_TEST_INTEGRATION=1   (set in shell before running)
    Redis on localhost:6379
    Postgres on localhost:5432 with octane DB

E2E tests additionally require:
    OCTANE_TEST_E2E=1
    Bodega Inference Engine on localhost:44468 with models loaded

Run integration only:
    OCTANE_TEST_INTEGRATION=1 python -m pytest tests/integration/ -v

Run integration + e2e:
    OCTANE_TEST_INTEGRATION=1 OCTANE_TEST_E2E=1 python -m pytest tests/integration/ tests/e2e/ -v
"""

from __future__ import annotations

import os
import pytest


pytestmark = pytest.mark.skipif(
    not os.getenv("OCTANE_TEST_INTEGRATION"),
    reason="Set OCTANE_TEST_INTEGRATION=1 to run integration tests",
)
