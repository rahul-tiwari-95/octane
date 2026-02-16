"""Preference Manager — CRUD on user preferences."""

from __future__ import annotations


class PreferenceManager:
    """Phase 1 stub. Full implementation in Session 5+."""

    async def get(self, user_id: str, key: str) -> str | None:
        return None

    async def set(self, user_id: str, key: str, value: str) -> None:
        pass

    async def get_all(self, user_id: str) -> dict[str, str]:
        return {}
