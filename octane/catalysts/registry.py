"""CatalystRegistry — keyword matching + structured data resolution.

Catalysts are keyed by name. Each entry declares:
  - triggers   : keywords that activate the catalyst (any match = candidate)
  - requires   : structural keys that must exist in upstream_results data
                 (Option B: match by data shape, not by agent name/type)
  - function   : the catalyst callable
  - sector     : informational label (finance / research / career / health / content)

Matching algorithm:
  1. Score each catalyst by counting trigger keyword hits in the query.
  2. Keep only candidates whose `requires` keys are satisfied by upstream data.
  3. Return the highest-scoring satisfied candidate (None if none qualify).
"""

from __future__ import annotations

import structlog
from typing import Any, Callable

logger = structlog.get_logger().bind(component="catalysts.registry")

# ── Lazy imports of catalyst functions ────────────────────────────────────────
# Imported at call time so that optional deps (matplotlib etc.) are only loaded
# when the catalyst actually runs, not on every CodeAgent instantiation.

def _load_price_chart():
    from octane.catalysts.finance.price_chart import price_chart
    return price_chart

def _load_return_calculator():
    from octane.catalysts.finance.return_calculator import return_calculator
    return return_calculator

def _load_portfolio_projection():
    from octane.catalysts.finance.portfolio_projection import portfolio_projection
    return portfolio_projection

def _load_technical_indicators():
    from octane.catalysts.finance.technical_indicators import technical_indicators
    return technical_indicators

def _load_allocation_pie():
    from octane.catalysts.finance.allocation_pie import allocation_pie
    return allocation_pie


# ── Registry definition ───────────────────────────────────────────────────────

CATALYST_REGISTRY: dict[str, dict] = {
    # ── Finance ───────────────────────────────────────────────────────────────
    "price_chart": {
        "sector": "finance",
        "triggers": [
            "chart", "plot", "graph", "price history", "visualize",
            "last month", "past month", "over time", "trend",
            "historical", "history", "timeseries", "time series",
        ],
        "requires": ["time_series"],          # key must exist in upstream data dict
        "loader": _load_price_chart,
    },
    "return_calculator": {
        "sector": "finance",
        "triggers": [
            "return", "p&l", "profit", "loss", "gain",
            "how much", "invested", "made", "lost", "worth",
        ],
        "requires": ["price", "regularMarketPrice"],  # any one of these
        "loader": _load_return_calculator,
    },
    "portfolio_projection": {
        "sector": "finance",
        "triggers": [
            "project", "simulate", "compound", "grow", "dca",
            "monte carlo", "retirement", "10 years", "20 years",
            "30 years", "long term", "future value",
        ],
        "requires": ["price", "regularMarketPrice"],
        "loader": _load_portfolio_projection,
    },
    "technical_indicators": {
        "sector": "finance",
        "triggers": [
            "rsi", "moving average", "macd", "bollinger",
            "technical analysis", "overbought", "oversold",
            "indicator", "signal",
        ],
        "requires": ["time_series"],
        "loader": _load_technical_indicators,
    },
    "allocation_pie": {
        "sector": "finance",
        "triggers": [
            "allocation", "diversification", "breakdown", "pie chart",
            "portfolio mix", "weighting", "percentage",
        ],
        "requires": ["price", "regularMarketPrice"],
        "loader": _load_allocation_pie,
    },
}


def _data_satisfies(requires: list[str], upstream_results: dict[str, Any]) -> bool:
    """Option B structural matching — check if any upstream result contains
    at least one of the required keys.

    Returns True if ANY dep's data dict contains ANY of the required keys.
    This avoids coupling to agent names or node IDs.
    """
    if not requires:
        return True
    for dep_data in upstream_results.values():
        if isinstance(dep_data, dict):
            if any(key in dep_data for key in requires):
                return True
    return False


def _resolve_upstream(requires: list[str], upstream_results: dict[str, Any]) -> dict[str, Any]:
    """Return the first upstream data dict that satisfies the requires list."""
    for dep_data in upstream_results.values():
        if isinstance(dep_data, dict):
            if any(key in dep_data for key in requires):
                return dep_data
    return {}


class CatalystRegistry:
    """Matches a query + upstream data to the best available catalyst.

    Usage:
        registry = CatalystRegistry()
        result = registry.match(query, upstream_results)
        if result:
            catalyst_fn, resolved_data = result
            response = catalyst_fn(resolved_data, output_dir, correlation_id)
    """

    def match(
        self,
        query: str,
        upstream_results: dict[str, Any],
    ) -> tuple[Callable, dict[str, Any]] | None:
        """Find the best catalyst for this query + upstream data.

        Returns:
            (catalyst_fn, resolved_data) if a match is found.
            None if no catalyst qualifies — fall through to LLM pipeline.
        """
        query_lower = query.lower()
        best_name: str | None = None
        best_score: int = 0

        for name, entry in CATALYST_REGISTRY.items():
            # Score by trigger keyword hits
            score = sum(1 for kw in entry["triggers"] if kw in query_lower)
            if score == 0:
                continue

            # Check structural data requirements (Option B)
            requires = entry.get("requires", [])
            if not _data_satisfies(requires, upstream_results):
                logger.debug(
                    "catalyst_data_missing",
                    catalyst=name,
                    requires=requires,
                    available_keys=_available_keys(upstream_results),
                )
                continue

            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None:
            return None

        entry = CATALYST_REGISTRY[best_name]
        resolved_data = _resolve_upstream(entry.get("requires", []), upstream_results)
        # Also pass the full upstream_results so the catalyst can access everything
        resolved_data = {**resolved_data, "_upstream": upstream_results}

        catalyst_fn = entry["loader"]()
        logger.info(
            "catalyst_matched",
            catalyst=best_name,
            score=best_score,
            sector=entry["sector"],
        )
        return catalyst_fn, resolved_data


def _available_keys(upstream_results: dict[str, Any]) -> list[str]:
    """Helper for debug logging — collect all keys present in upstream data."""
    keys: list[str] = []
    for dep_data in upstream_results.values():
        if isinstance(dep_data, dict):
            keys.extend(dep_data.keys())
    return list(set(keys))
