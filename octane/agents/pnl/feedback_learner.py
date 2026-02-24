"""Feedback Learner — records explicit and implicit feedback signals,
then nudges user preferences based on accumulated evidence.

Signal types
------------
thumbs_up   : +1  (user liked the response)
thumbs_down : -1  (user disliked the response)
time_spent  : float seconds the user spent reading (longer → positive signal)

Key scheme
----------
pnl:{user_id}:fb:{correlation_id}   → JSON payload (single signal)
pnl:{user_id}:fb:score              → running integer score (incr/decr)

Nudge logic
-----------
Every NUDGE_THRESHOLD signals the learner recalculates aggregate score and
adjusts preferences:
  score ≥ +NUDGE_THRESHOLD → shift verbosity toward "detailed"
  score ≤ -NUDGE_THRESHOLD → shift verbosity toward "concise"
Score is then reset to 0 so it doesn't drift indefinitely.
"""

from __future__ import annotations

import json
import time

import structlog

from octane.agents.pnl.preference_manager import PreferenceManager
from octane.tools.redis_client import RedisClient

logger = structlog.get_logger().bind(component="pnl.feedback")

NUDGE_THRESHOLD = 3  # signals before a preference nudge fires

# Verbosity ladder — nudges move one step at a time
_VERBOSITY_LADDER = ["concise", "balanced", "detailed"]

# time_spent → rough score: <5s negative, 5-20s neutral, >20s positive
_TIME_BREAKPOINTS = [(5.0, -1), (20.0, 0), (float("inf"), 1)]


def _time_score(seconds: float) -> int:
    for threshold, score in _TIME_BREAKPOINTS:
        if seconds < threshold:
            return score
    return 1


class FeedbackLearner:
    """Records user feedback and nudges preferences when evidence accumulates."""

    def __init__(
        self,
        redis: RedisClient | None = None,
        prefs: PreferenceManager | None = None,
    ) -> None:
        self._redis = redis or RedisClient()
        self._prefs = prefs or PreferenceManager(redis=self._redis)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def record(
        self,
        user_id: str,
        signal_type: str,
        value: float,
        correlation_id: str | None = None,
    ) -> None:
        """Store a signal and trigger a preference nudge if threshold is reached."""
        score_delta = self._signal_to_delta(signal_type, value)
        if score_delta == 0:
            logger.debug("feedback_neutral", user_id=user_id, signal_type=signal_type)
            return

        # Persist individual signal
        payload = json.dumps({
            "signal_type": signal_type,
            "value": value,
            "delta": score_delta,
            "ts": time.time(),
        })
        cid = correlation_id or f"{int(time.time() * 1000)}"
        await self._redis.set(f"pnl:{user_id}:fb:{cid}", payload, ttl=60 * 60 * 24 * 30)
        logger.info("feedback_recorded", user_id=user_id, signal=signal_type,
                    delta=score_delta, cid=cid)

        # Update running score and maybe nudge
        new_score = await self._update_score(user_id, score_delta)
        logger.debug("feedback_score", user_id=user_id, score=new_score)

        if abs(new_score) >= NUDGE_THRESHOLD:
            await self._nudge(user_id, new_score)
            # Reset score after nudge
            await self._redis.set(f"pnl:{user_id}:fb:score", "0", ttl=0)

    async def get_score(self, user_id: str) -> int:
        """Return the current running score for a user."""
        raw = await self._redis.get(f"pnl:{user_id}:fb:score")
        return int(raw) if raw else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _signal_to_delta(self, signal_type: str, value: float) -> int:
        if signal_type == "thumbs_up":
            return 1
        if signal_type == "thumbs_down":
            return -1
        if signal_type == "time_spent":
            return _time_score(value)
        logger.warning("feedback_unknown_signal", signal_type=signal_type)
        return 0

    async def _update_score(self, user_id: str, delta: int) -> int:
        score_key = f"pnl:{user_id}:fb:score"
        current = await self._redis.get(score_key)
        new_score = (int(current) if current else 0) + delta
        await self._redis.set(score_key, str(new_score), ttl=0)
        return new_score

    async def _nudge(self, user_id: str, score: int) -> None:
        """Shift verbosity one step in the direction of the score."""
        current = await self._prefs.get(user_id, "verbosity")
        try:
            idx = _VERBOSITY_LADDER.index(current)
        except ValueError:
            idx = 0  # default to first step if unknown

        if score > 0:
            new_idx = min(idx + 1, len(_VERBOSITY_LADDER) - 1)
        else:
            new_idx = max(idx - 1, 0)

        new_verbosity = _VERBOSITY_LADDER[new_idx]
        if new_verbosity != current:
            await self._prefs.set(user_id, "verbosity", new_verbosity)
            logger.info("feedback_nudge_verbosity", user_id=user_id,
                        old=current, new=new_verbosity, score=score)
        else:
            logger.debug("feedback_nudge_noop", user_id=user_id,
                         verbosity=current, score=score)
