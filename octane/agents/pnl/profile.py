"""Profile — aggregated user profile from all signals."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Aggregated user profile."""

    user_id: str = "default"
    expertise_level: str = "advanced"
    preferred_verbosity: str = "concise"
    domains: list[str] = Field(default_factory=lambda: ["technology", "finance"])
    last_active: datetime | None = None

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.now(timezone.utc)
