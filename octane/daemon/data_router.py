"""Data Router — algorithmic Redis vs Postgres placement engine.

This is NOT an LLM call. This is pure CS — the same kind of logic an OS
uses to decide whether a page belongs in RAM or on disk.

Every piece of data Octane produces belongs to a DataCategory.  Each category
has a pre-defined PlacementPolicy that determines:
    1. Which storage backend (Redis, Postgres, or Redis-with-promotion)
    2. Which TTL tier governs its lifetime
    3. What key prefix/namespace to use in Redis
    4. What Postgres table to target (if applicable)

The DataRouter applies these policies to route store/retrieve/delete operations.
It also runs a periodic promotion sweep — data that proves its worth in Redis
(high access count or long survival) gets persisted to Postgres.

Key hashing: All Redis keys are prefixed with a namespace derived from
DataCategory.  This gives us:
    - Clean separation of data types in Redis
    - Ability to purge entire categories (e.g., all expired search results)
    - Predictable key patterns for monitoring

Example keys:
    qcache:sha256_prefix:8chars   — query dedup cache
    sess:session_id               — active session state
    search:query_hash             — recent search results
    msr:session:question_hash     — MSR question/answer pairs
    rlimit:endpoint:window        — rate limit counters
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any

import structlog

from octane.daemon.cache_policy import (
    CacheEntry,
    CachePolicyEngine,
    PromotionPolicy,
    TTLTier,
)

logger = structlog.get_logger().bind(component="data_router")


# ── Storage Backend ───────────────────────────────────────────────────────────


class StorageBackend(str, enum.Enum):
    """Where data physically lives."""

    REDIS = "redis"                         # Hot cache only
    POSTGRES = "postgres"                   # Structured durable storage
    REDIS_WITH_PROMOTION = "redis_promote"  # Start in Redis, graduate to PG


# ── Data Categories ───────────────────────────────────────────────────────────


class DataCategory(str, enum.Enum):
    """Classification of every data type Octane produces.

    Each category maps to exactly one PlacementPolicy.  The mapping is
    STATIC and DETERMINISTIC — no LLM reasoning needed.  The rules are
    derived from first principles:

        - How often is this data read? (access frequency)
        - How quickly does it become stale? (lifespan)
        - Does it need relational queries? (query complexity)
        - How large is it? (size)
        - Does it need to survive daemon restarts? (durability)
    """

    # ── Redis-only (ephemeral, high-frequency) ──
    QUERY_DEDUP = "query_dedup"             # Prevent duplicate queries within window
    RATE_LIMIT = "rate_limit"               # API rate limit counters
    MODEL_STATUS = "model_status"           # Which models are loaded right now
    HEALTH_CHECK = "health_check"           # Last known health of each service

    # ── Redis with promotion ──
    SESSION_STATE = "session_state"         # Active conversation state
    SEARCH_RESULTS = "search_results"       # Recent web search results
    MSR_CONTEXT = "msr_context"             # MSR question/answer pairs
    AGENT_RESULT = "agent_result"           # Individual agent response cache

    # ── Postgres-only (durable, relational) ──
    RESEARCH_REPORT = "research_report"     # Completed research output
    USER_PREFERENCE = "user_preference"     # P&L preferences
    MEMORY_CHUNK = "memory_chunk"           # Long-term memory storage
    SESSION_HISTORY = "session_history"     # Completed session transcripts
    CATALYST_DEF = "catalyst_def"           # Catalyst definitions
    MONITOR_CONFIG = "monitor_config"       # Monitor/shadow configurations


# ── Key Namespace Prefixes ────────────────────────────────────────────────────

_KEY_PREFIX: dict[DataCategory, str] = {
    DataCategory.QUERY_DEDUP: "qcache",
    DataCategory.RATE_LIMIT: "rlimit",
    DataCategory.MODEL_STATUS: "mstat",
    DataCategory.HEALTH_CHECK: "hcheck",
    DataCategory.SESSION_STATE: "sess",
    DataCategory.SEARCH_RESULTS: "search",
    DataCategory.MSR_CONTEXT: "msr",
    DataCategory.AGENT_RESULT: "ares",
    DataCategory.RESEARCH_REPORT: "report",
    DataCategory.USER_PREFERENCE: "upref",
    DataCategory.MEMORY_CHUNK: "memchk",
    DataCategory.SESSION_HISTORY: "shist",
    DataCategory.CATALYST_DEF: "catdef",
    DataCategory.MONITOR_CONFIG: "monconf",
}


# ── Placement Policy ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlacementPolicy:
    """Static routing rule for a DataCategory.

    Each category has exactly one policy. The policy is immutable — it's
    defined at startup and never changes. This is a compile-time decision,
    not a runtime one.

    Attributes:
        backend:        Where to store (Redis / Postgres / Redis+Promotion).
        ttl_tier:       TTL tier for cache entries.
        key_prefix:     Redis key namespace prefix.
        pg_table:       Postgres table name (None for Redis-only categories).
        pinned:         If True, entries are pinned in cache (never LRU-evicted).
        max_value_bytes: Soft cap on individual value size. Values exceeding
                         this are routed to Postgres regardless of policy.
    """

    backend: StorageBackend
    ttl_tier: TTLTier
    key_prefix: str
    pg_table: str | None = None
    pinned: bool = False
    max_value_bytes: int = 65_536  # 64 KB default


# ── Policy Registry (static, deterministic) ──────────────────────────────────

PLACEMENT_POLICIES: dict[DataCategory, PlacementPolicy] = {
    # ── Redis-only: ephemeral, high-frequency ──
    DataCategory.QUERY_DEDUP: PlacementPolicy(
        backend=StorageBackend.REDIS,
        ttl_tier=TTLTier.EPHEMERAL,
        key_prefix="qcache",
        max_value_bytes=4_096,
    ),
    DataCategory.RATE_LIMIT: PlacementPolicy(
        backend=StorageBackend.REDIS,
        ttl_tier=TTLTier.EPHEMERAL,
        key_prefix="rlimit",
        max_value_bytes=256,
    ),
    DataCategory.MODEL_STATUS: PlacementPolicy(
        backend=StorageBackend.REDIS,
        ttl_tier=TTLTier.SESSION,
        key_prefix="mstat",
        pinned=True,  # Never evict — always need to know what's loaded
        max_value_bytes=2_048,
    ),
    DataCategory.HEALTH_CHECK: PlacementPolicy(
        backend=StorageBackend.REDIS,
        ttl_tier=TTLTier.EPHEMERAL,
        key_prefix="hcheck",
        max_value_bytes=1_024,
    ),

    # ── Redis with promotion: start hot, graduate to durable ──
    DataCategory.SESSION_STATE: PlacementPolicy(
        backend=StorageBackend.REDIS_WITH_PROMOTION,
        ttl_tier=TTLTier.SESSION,
        key_prefix="sess",
        pg_table="session_history",
        pinned=True,  # Active sessions must never be evicted
        max_value_bytes=131_072,  # 128 KB — sessions can be large
    ),
    DataCategory.SEARCH_RESULTS: PlacementPolicy(
        backend=StorageBackend.REDIS_WITH_PROMOTION,
        ttl_tier=TTLTier.WARM,
        key_prefix="search",
        pg_table="web_pages",
        max_value_bytes=262_144,  # 256 KB — full page content
    ),
    DataCategory.MSR_CONTEXT: PlacementPolicy(
        backend=StorageBackend.REDIS_WITH_PROMOTION,
        ttl_tier=TTLTier.SESSION,
        key_prefix="msr",
        pg_table="memory_chunks",
        max_value_bytes=32_768,
    ),
    DataCategory.AGENT_RESULT: PlacementPolicy(
        backend=StorageBackend.REDIS_WITH_PROMOTION,
        ttl_tier=TTLTier.WARM,
        key_prefix="ares",
        pg_table="memory_chunks",
        max_value_bytes=131_072,
    ),

    # ── Postgres-only: durable, relational ──
    DataCategory.RESEARCH_REPORT: PlacementPolicy(
        backend=StorageBackend.POSTGRES,
        ttl_tier=TTLTier.PERSISTENT,
        key_prefix="report",
        pg_table="research_reports",
    ),
    DataCategory.USER_PREFERENCE: PlacementPolicy(
        backend=StorageBackend.POSTGRES,
        ttl_tier=TTLTier.PERSISTENT,
        key_prefix="upref",
        pg_table="user_preferences",
    ),
    DataCategory.MEMORY_CHUNK: PlacementPolicy(
        backend=StorageBackend.POSTGRES,
        ttl_tier=TTLTier.PERSISTENT,
        key_prefix="memchk",
        pg_table="memory_chunks",
    ),
    DataCategory.SESSION_HISTORY: PlacementPolicy(
        backend=StorageBackend.POSTGRES,
        ttl_tier=TTLTier.PERSISTENT,
        key_prefix="shist",
        pg_table="session_history",
    ),
    DataCategory.CATALYST_DEF: PlacementPolicy(
        backend=StorageBackend.POSTGRES,
        ttl_tier=TTLTier.PERSISTENT,
        key_prefix="catdef",
        pg_table="catalysts",
    ),
    DataCategory.MONITOR_CONFIG: PlacementPolicy(
        backend=StorageBackend.POSTGRES,
        ttl_tier=TTLTier.PERSISTENT,
        key_prefix="monconf",
        pg_table="monitor_configs",
    ),
}


def get_policy(category: DataCategory) -> PlacementPolicy:
    """Look up the placement policy for a data category.

    Raises KeyError if category has no registered policy (indicates a bug).
    """
    return PLACEMENT_POLICIES[category]


# ── Key Hashing ───────────────────────────────────────────────────────────────


def make_key(category: DataCategory, *parts: str) -> str:
    """Build a namespaced Redis key from category + arbitrary parts.

    Format: prefix:part1:part2:...

    The prefix comes from the category's placement policy.
    Parts are joined with ':'.  If a part looks like a long string
    (>64 chars), it's hashed to keep keys short.

    Examples:
        make_key(QUERY_DEDUP, "what is octane")
        → "qcache:what is octane"

        make_key(SEARCH_RESULTS, session_id, query_hash)
        → "search:abc123:7f3a9b..."

        make_key(SESSION_STATE, very_long_session_id)
        → "sess:sha256_first_16_chars"
    """
    policy = get_policy(category)
    processed: list[str] = [policy.key_prefix]

    for part in parts:
        if len(part) > 64:
            # Hash long parts to keep Redis keys compact
            processed.append(hash_key_part(part))
        else:
            processed.append(part)

    return ":".join(processed)


def hash_key_part(value: str) -> str:
    """SHA-256 hash a string and return first 16 hex chars.

    Collision probability at 16 hex chars (64 bits): ~1 in 2^64.
    For Octane's scale (thousands, not billions of keys), this is safe.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


# ── Routing Decision ─────────────────────────────────────────────────────────


@dataclass
class RoutingDecision:
    """The output of route() — tells the caller exactly where to put data.

    Attributes:
        backend:     Where to store (REDIS, POSTGRES, or both).
        redis_key:   Full Redis key (None if Postgres-only).
        pg_table:    Postgres table (None if Redis-only).
        ttl_seconds: TTL to set in Redis (0 = no expiry).
        pinned:      Whether to pin in cache policy.
        category:    Original data category.
        overflow:    True if value exceeded max_value_bytes → forced to PG.
    """

    backend: StorageBackend
    redis_key: str | None
    pg_table: str | None
    ttl_seconds: float
    pinned: bool
    category: DataCategory
    overflow: bool = False


class DataRouter:
    """Routes data to the correct storage backend.

    The DataRouter is the single point of truth for "where does this data go?"
    It wraps the CachePolicyEngine for LRU/TTL tracking and applies
    PlacementPolicies to make deterministic routing decisions.

    It does NOT perform actual I/O — it returns RoutingDecisions that the
    PoolManager or store clients execute.

    Args:
        cache_engine:       CachePolicyEngine for LRU/TTL management.
        promotion_policy:   Override default promotion thresholds.
    """

    def __init__(
        self,
        cache_engine: CachePolicyEngine | None = None,
        promotion_policy: PromotionPolicy | None = None,
    ) -> None:
        promo = promotion_policy or PromotionPolicy()
        self.cache = cache_engine or CachePolicyEngine(promotion_policy=promo)

    def route(
        self,
        category: DataCategory,
        *key_parts: str,
        value_bytes: int = 0,
    ) -> RoutingDecision:
        """Determine where a piece of data should be stored.

        This is the core routing function. It applies the static placement
        policy and checks for overflow (value too large for Redis).

        Args:
            category:    What kind of data this is.
            key_parts:   Parts to build the Redis key from.
            value_bytes: Size of the value in bytes.

        Returns:
            RoutingDecision with all information needed to execute the store.
        """
        policy = get_policy(category)
        redis_key = make_key(category, *key_parts)

        from octane.daemon.cache_policy import TTL_BASE
        ttl = TTL_BASE[policy.ttl_tier]

        # Overflow: value exceeds max_value_bytes → force to Postgres
        if (
            value_bytes > policy.max_value_bytes
            and policy.backend != StorageBackend.POSTGRES
        ):
            logger.debug(
                "data_overflow_to_postgres",
                category=category.value,
                value_bytes=value_bytes,
                max_bytes=policy.max_value_bytes,
            )
            return RoutingDecision(
                backend=StorageBackend.POSTGRES,
                redis_key=None,
                pg_table=policy.pg_table or "memory_chunks",
                ttl_seconds=0,
                pinned=False,
                category=category,
                overflow=True,
            )

        # Standard routing based on policy
        return RoutingDecision(
            backend=policy.backend,
            redis_key=redis_key if policy.backend != StorageBackend.POSTGRES else None,
            pg_table=policy.pg_table,
            ttl_seconds=ttl,
            pinned=policy.pinned,
            category=category,
        )

    def admit(
        self,
        category: DataCategory,
        *key_parts: str,
        value_bytes: int = 0,
    ) -> tuple[RoutingDecision, list[str]]:
        """Route AND register in cache policy engine.

        Combines route() with cache admission. Returns the routing decision
        and a list of any keys evicted from cache to make room.

        This is the primary method for "I have data, where does it go and
        what did we have to evict?"
        """
        decision = self.route(category, *key_parts, value_bytes=value_bytes)

        evicted: list[str] = []

        # Only track in cache engine if data goes to Redis
        if decision.backend in (
            StorageBackend.REDIS,
            StorageBackend.REDIS_WITH_PROMOTION,
        ):
            redis_key = decision.redis_key
            if redis_key:
                policy = get_policy(category)
                evicted = self.cache.admit(
                    key=redis_key,
                    tier=policy.ttl_tier,
                    size_bytes=value_bytes,
                    pinned=policy.pinned,
                )

        return decision, evicted

    def record_access(self, category: DataCategory, *key_parts: str) -> CacheEntry | None:
        """Record a read access to cached data.

        Updates LRU position and TTL. Returns the cache entry or None
        if the key isn't tracked (e.g., Postgres-only data).
        """
        policy = get_policy(category)
        if policy.backend == StorageBackend.POSTGRES:
            return None

        redis_key = make_key(category, *key_parts)
        return self.cache.access(redis_key)

    def get_promotion_candidates(self) -> list[str]:
        """Return Redis keys eligible for promotion to Postgres.

        These are entries that have been accessed frequently enough or
        survived long enough to warrant durable storage.
        """
        return self.cache.get_promotion_candidates()

    def mark_promoted(self, category: DataCategory, *key_parts: str) -> bool:
        """Mark a key as promoted (already persisted to Postgres)."""
        redis_key = make_key(category, *key_parts)
        return self.cache.mark_promoted(redis_key)

    def sweep_expired(self) -> list[str]:
        """Run garbage collection — remove all expired entries."""
        return self.cache.evict_expired()

    def snapshot(self) -> dict[str, Any]:
        """Cache state snapshot for monitoring."""
        return self.cache.snapshot()
