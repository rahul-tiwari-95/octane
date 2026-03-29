"""Local-only session auth for Octane UI.

No cloud, no OAuth. Just a random token generated on first launch,
stored at ~/.octane/ui_token. Browser sends it via cookie.
"""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["auth"])

_TOKEN_PATH = Path.home() / ".octane" / "ui_token"


def get_or_create_token() -> str:
    """Get existing UI token or generate a new one."""
    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _TOKEN_PATH.exists():
        return _TOKEN_PATH.read_text().strip()
    token = secrets.token_urlsafe(32)
    _TOKEN_PATH.write_text(token)
    _TOKEN_PATH.chmod(0o600)
    return token


def verify_token(request: Request) -> bool:
    """Check if the request has a valid session token.

    For local development (localhost / 127.0.0.1), auth is skipped.
    """
    host = request.client.host if request.client else "127.0.0.1"
    if host in ("127.0.0.1", "::1", "localhost"):
        return True
    token = request.cookies.get("octane_session")
    if not token:
        return False
    return secrets.compare_digest(token, get_or_create_token())


@router.post("/auth/token")
async def get_token(response: Response) -> dict[str, str]:
    """Issue a session cookie with the local auth token."""
    token = get_or_create_token()
    response.set_cookie(
        key="octane_session",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 365,  # 1 year
    )
    return {"status": "ok"}
