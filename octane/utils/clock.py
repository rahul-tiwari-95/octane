"""Centralised wall-clock helpers — single source of truth for 'now'.

Every part of Octane that needs to know the current date imports from here.
This prevents drift where different modules call datetime.now() at slightly
different times, and makes test-time mocking trivial (patch one function).

Usage:
    from octane.utils.clock import today_str, today_human, month_year, now_utc
"""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def today_str() -> str:
    """ISO 8601 date string: '2026-02-23'"""
    return now_utc().strftime("%Y-%m-%d")


def today_human() -> str:
    """Full human-readable date: 'Monday, February 23, 2026'"""
    return now_utc().strftime("%A, %B %-d, %Y")


def month_year() -> str:
    """Month + year for search query enrichment: 'February 2026'"""
    return now_utc().strftime("%B %Y")


def year() -> str:
    """Four-digit year string: '2026'"""
    return now_utc().strftime("%Y")
