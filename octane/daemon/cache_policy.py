"""Cache policy engine — LRU eviction, TTL management, promotion tracking.

Pure algorithmic — no I/O, no network, no Redis/Postgres calls.
This module defines the RULES for how data lives and dies in cache.
The DataRouter (data_router.py) applies these rules against real stores.

Design principles (OS scheduling theory applied to cache):
    1. LRU eviction   — Least Recently Used, like page replacement in a VM.
    2. TTL tiers       — Data has a natural lifespan. Ephemeral (5m) for
                         query dedup; Session (2h) for active work; Warm (24h)
                         for recent results; Persistent for pinned data.
    3. Sliding window  — Accessing data before expiry extends its TTL, up to
                         a max. Hot data stays alive; cold data dies on schedule.
    4. Pinning         — Critical data (active session state, loaded model
                         registry) is pinned and NEVER evicted by LRU.
    5. Promotion       — Data that survives long enough in Redis (hot cache)
                         gets promoted to Postgres (warm/cold storage). Like
                         a page being marked as "frequently accessed" and moved
                         to a faster tier — except here Redis IS the fast tier
                         and Postgres is the durable tier.

All timestamps are monotonic (time.monotonic) for correctness — wall clock
can jump on sleep/wake, monotonic never does.
"""

from __future__ import annotations

import enum
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


# ── TTL Tiers ─────────────────────────────────────────────────────────────────


class TTLTier(str, enum.Enum):
    """How long data should live in cache before expiring.

    Each tier defines a base TTL and a max TTL (for sliding window extension).
    These are modelled after OS scheduling quanta — short-lived processes get
    small time slices, long-lived processes get larger ones.
    """

    EPHEMERAL = "ephemeral"    # 5 min base, 15 min max  — query dedup, rate limits
    SESSION = "session"        # 2 hour base, 6 hour max — active session state
    WARM = "warm"              # 24 hour base, 72 hour max — recent search results
    PERSISTENT = "persistent"  # No TTL — pinned, only evicted explicitly


# TTL values in seconds
TTL_BASE: dict[TTLTier, float] = {
    TTLTier.EPHEMERAL: 300.0,       # 5 min
    TTLTier.SESSION: 7200.0,        # 2 hours
    TTLTier.WARM: 86400.0,          # 24 hours
    TTLTier.PERSISTENT: 0.0,        # 0 = no expiry
}

TTL_MAX: dict[TTLTier, float] = {
    TTLTier.EPHEMERAL: 900.0,       # 15 min
    TTLTier.SESSION: 21600.0,       # 6 hours
    TTLTier.WARM: 259200.0,         # 72 hours
    TTLTier.PERSISTENT: 0.0,        # no expiry
}


# ── Cache Entry ───────────────────────────────────────────────────────────────


@dataclass
class CacheEntry:
    """A single item tracked by the cache policy engine.

    Tracks all metadata needed for LRU eviction, TTL management, and
    promotion decisions. The actual VALUE is NOT stored here — the policy
    engine only tracks metadata. Values live in Redis/Postgres.

    Attributes:
        key:            Unique cache key (namespaced, e.g. "query_cache:abc123").
        tier:           TTL tier controlling base/max lifetimes.
        created_at:     Monotonic timestamp when entry was first created.
        last_accessed:  Monotonic timestamp of most recent read/write.
        expire_at:      Monotonic timestamp when this entry expires (0 = never).
        access_count:   Total number of reads since creation.
        size_bytes:     Approximate size of the cached value in bytes.
        pinned:         If True, entry is immune to LRU eviction.
        promoted:       If True, entry has been promoted to Postgres.
    """

    key: str
    tier: TTLTier
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    expire_at: float = 0.0
    access_count: int = 0
    size_bytes: int = 0
    pinned: bool = False
    promoted: bool = False

    def is_expired(self, now: float | None = None) -> bool:
        """Check if this entry has exceeded its TTL.

        Persistent entries (expire_at == 0) never expire via TTL.
        """
        if self.expire_at == 0.0:
            return False
        if now is None:
            now = time.monotonic()
        return now >= self.expire_at

    def touch(self, now: float | None = None) -> None:
        """Record an access — updates last_accessed, increments count,
        and extends TTL via sliding window.

        Sliding window: each access pushes expire_at forward by base_ttl
        from current time, but never beyond (created_at + max_ttl).
        This means frequently accessed data stays alive longer, but
        nothing lives forever unless pinned.
        """
        if now is None:
            now = time.monotonic()
        self.last_accessed = now
        self.access_count += 1

        # Extend TTL (sliding window) — but not for persistent entries
        if self.tier == TTLTier.PERSISTENT:
            return

        base = TTL_BASE[self.tier]
        max_ttl = TTL_MAX[self.tier]
        new_expire = now + base
        # Cap at created_at + max_ttl
        ceiling = self.created_at + max_ttl
        self.expire_at = min(new_expire, ceiling)

    @property
    def age(self) -> float:
        """Seconds since creation (monotonic)."""
        return time.monotonic() - self.created_at


# ── Promotion Policy ──────────────────────────────────────────────────────────


@dataclass
class PromotionPolicy:
    """Rules for when Redis data should be promoted to Postgres.

    Promotion is the cache equivalent of "this page is accessed so often
    it should be pinned in physical memory." Here, data that proves its
    worth in the fast-but-volatile Redis tier gets persisted to Postgres.

    An entry is promotion-eligible when ANY of these thresholds is met:
        - access_count >= min_access_count  (frequently read)
        - age >= min_age_seconds            (long-lived)

    Both can be configured per use case. The DataRouter checks these
    and triggers the actual Redis→Postgres write.
    """

    min_access_count: int = 10          # Promote after 10 reads
    min_age_seconds: float = 3600.0     # Or promote after 1 hour alive

    def should_promote(self, entry: CacheEntry, now: float | None = None) -> bool:
        """Check if an entry qualifies for promotion to Postgres.

        Returns False for:
            - Already promoted entries
            - Pinned entries (they stay in Redis by design)
            - EPHEMERAL entries (they die quickly, not worth persisting)

        Args:
            entry: The cache entry to evaluate.
            now:   Monotonic timestamp for age calculation. If None,
                   uses time.monotonic(). Must be passed for testability.
        """
        if entry.promoted:
            return False
        if entry.pinned:
            return False
        if entry.tier == TTLTier.EPHEMERAL:
            return False

        if now is None:
            now = time.monotonic()
        age = now - entry.created_at

        if entry.access_count >= self.min_access_count:
            return True
        if age >= self.min_age_seconds:
            return True
        return False


# ── LRU Cache Policy Engine ──────────────────────────────────────────────────


class CachePolicyEngine:
    """LRU eviction engine with TTL tiers, pinning, and promotion.

    This is the brain of Octane's cache. It decides:
        - What to keep (recently used, pinned)
        - What to evict (least recently used, expired)
        - What to promote (frequently accessed → Postgres)

    The engine tracks METADATA only — it never touches actual data stores.
    The DataRouter calls engine methods and then applies decisions to Redis/PG.

    Implementation: OrderedDict for O(1) LRU operations.
    move_to_end() on access, popitem(last=False) for eviction.

    Args:
        max_entries:      Maximum number of tracked entries before eviction.
        max_memory_bytes: Soft memory budget. When total tracked size exceeds
                          this, LRU eviction kicks in even if under max_entries.
                          0 = no memory-based eviction (count-only).
        promotion_policy: Rules for Redis → Postgres promotion.
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        max_memory_bytes: int = 0,
        promotion_policy: PromotionPolicy | None = None,
    ) -> None:
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self._total_bytes: int = 0

        # Counters for observability
        self.stats = CacheStats()

    @property
    def size(self) -> int:
        """Number of tracked entries."""
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        """Total tracked byte size across all entries."""
        return self._total_bytes

    def admit(
        self,
        key: str,
        tier: TTLTier,
        size_bytes: int = 0,
        pinned: bool = False,
        now: float | None = None,
    ) -> list[str]:
        """Register a new cache entry. Returns list of evicted keys.

        If the key already exists, updates its metadata (re-admission).
        If capacity is exceeded, evicts LRU entries until there's room.
        Pinned entries are never evicted.

        Args:
            key:        Unique cache key.
            tier:       TTL tier for this entry.
            size_bytes: Approximate size of the cached value.
            pinned:     If True, immune to LRU eviction.
            now:        Current monotonic time (for testing).

        Returns:
            List of keys that were evicted to make room.
        """
        if now is None:
            now = time.monotonic()

        evicted: list[str] = []

        # Re-admission: remove old entry first
        if key in self._entries:
            old = self._entries.pop(key)
            self._total_bytes -= old.size_bytes

        # Create new entry
        entry = CacheEntry(
            key=key,
            tier=tier,
            created_at=now,
            last_accessed=now,
            size_bytes=size_bytes,
            pinned=pinned,
        )

        # Set initial TTL
        if tier != TTLTier.PERSISTENT:
            entry.expire_at = now + TTL_BASE[tier]
        # else: expire_at stays 0.0 (never expires)

        # Evict expired entries first (garbage collection pass)
        evicted.extend(self._evict_expired(now))

        # Evict LRU entries if over capacity
        evicted.extend(self._evict_lru(extra_bytes=size_bytes))

        # Insert at end (most recently used)
        self._entries[key] = entry
        self._total_bytes += size_bytes

        self.stats.admissions += 1
        return evicted

    def access(self, key: str, now: float | None = None) -> CacheEntry | None:
        """Record an access to a cache entry. Returns the entry or None.

        Updates LRU position (move to end), extends TTL via sliding window,
        increments access count. This is the equivalent of a "page hit" in
        OS virtual memory — the page gets moved to the front of the
        clock/LRU list.

        Returns None if the key doesn't exist or has expired.
        """
        if now is None:
            now = time.monotonic()

        entry = self._entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None

        # Check TTL expiry
        if entry.is_expired(now):
            self._remove(key)
            self.stats.expirations += 1
            self.stats.misses += 1
            return None

        # Update LRU position and access metadata
        self._entries.move_to_end(key)
        entry.touch(now)
        self.stats.hits += 1

        return entry

    def peek(self, key: str, now: float | None = None) -> CacheEntry | None:
        """Read entry metadata WITHOUT updating LRU position or access count.

        Useful for checking promotion eligibility without side effects.
        Returns None if missing or expired.
        """
        if now is None:
            now = time.monotonic()

        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_expired(now):
            self._remove(key)
            self.stats.expirations += 1
            return None
        return entry

    def remove(self, key: str) -> bool:
        """Explicitly remove an entry. Returns True if it existed."""
        if key in self._entries:
            self._remove(key)
            return True
        return False

    def mark_promoted(self, key: str) -> bool:
        """Mark an entry as promoted to Postgres. Returns True if found."""
        entry = self._entries.get(key)
        if entry is None:
            return False
        entry.promoted = True
        self.stats.promotions += 1
        return True

    def get_promotion_candidates(self, now: float | None = None) -> list[str]:
        """Return keys eligible for promotion to Postgres.

        Scans all entries and checks against PromotionPolicy thresholds.
        Called periodically by the DataRouter's promotion sweep.
        """
        if now is None:
            now = time.monotonic()

        candidates: list[str] = []
        for key, entry in self._entries.items():
            if entry.is_expired(now):
                continue
            if self.promotion_policy.should_promote(entry, now=now):
                candidates.append(key)
        return candidates

    def evict_expired(self, now: float | None = None) -> list[str]:
        """Public method to sweep and evict all expired entries.

        Returns list of evicted keys. Called by background maintenance tasks.
        """
        if now is None:
            now = time.monotonic()
        return self._evict_expired(now)

    def get_lru_order(self) -> list[str]:
        """Return keys in LRU order (least recently used first).

        Useful for debugging and testing. Does NOT include expired entries.
        """
        now = time.monotonic()
        return [
            k for k, e in self._entries.items()
            if not e.is_expired(now)
        ]

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of cache state for monitoring."""
        now = time.monotonic()
        live = [e for e in self._entries.values() if not e.is_expired(now)]
        pinned = [e for e in live if e.pinned]
        promoted = [e for e in live if e.promoted]

        return {
            "total_entries": len(live),
            "total_bytes": sum(e.size_bytes for e in live),
            "pinned_count": len(pinned),
            "promoted_count": len(promoted),
            "max_entries": self.max_entries,
            "max_memory_bytes": self.max_memory_bytes,
            "stats": self.stats.to_dict(),
            "tier_distribution": self._tier_distribution(live),
        }

    # ── Internal methods ──────────────────────────────────────────────────

    def _remove(self, key: str) -> None:
        """Remove entry and update byte counter."""
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry.size_bytes

    def _evict_expired(self, now: float) -> list[str]:
        """Remove all entries past their TTL. Returns evicted keys."""
        evicted: list[str] = []
        # Iterate over a snapshot to avoid mutation during iteration
        for key in list(self._entries.keys()):
            entry = self._entries[key]
            if entry.is_expired(now):
                self._remove(key)
                evicted.append(key)
                self.stats.expirations += 1
        return evicted

    def _evict_lru(self, extra_bytes: int = 0) -> list[str]:
        """Evict least-recently-used (unpinned) entries until capacity is met.

        Eviction triggers when:
            1. len(entries) >= max_entries, OR
            2. total_bytes + extra_bytes > max_memory_bytes (when budget > 0)

        Pinned entries are NEVER evicted — they're skipped.
        OrderedDict iteration starts from the oldest (LRU) end.

        Returns list of evicted keys.
        """
        evicted: list[str] = []

        while self._should_evict(extra_bytes):
            # Find the least-recently-used unpinned entry
            victim_key = self._find_lru_victim()
            if victim_key is None:
                break  # All remaining entries are pinned — can't evict
            self._remove(victim_key)
            evicted.append(victim_key)
            self.stats.evictions += 1

        return evicted

    def _should_evict(self, extra_bytes: int = 0) -> bool:
        """Check if eviction is needed based on count or memory."""
        if len(self._entries) >= self.max_entries:
            return True
        if (
            self.max_memory_bytes > 0
            and self._total_bytes + extra_bytes > self.max_memory_bytes
        ):
            return True
        return False

    def _find_lru_victim(self) -> str | None:
        """Find the least-recently-used non-pinned entry.

        OrderedDict preserves insertion/access order. Iteration starts
        from the LRU end (first item). We skip pinned entries.
        """
        for key, entry in self._entries.items():
            if not entry.pinned:
                return key
        return None  # All entries are pinned

    @staticmethod
    def _tier_distribution(entries: list[CacheEntry]) -> dict[str, int]:
        """Count entries per TTL tier."""
        dist: dict[str, int] = {}
        for entry in entries:
            tier_name = entry.tier.value
            dist[tier_name] = dist.get(tier_name, 0) + 1
        return dist


# ── Cache Stats ───────────────────────────────────────────────────────────────


@dataclass
class CacheStats:
    """Counters for cache observability.

    Hit rate = hits / (hits + misses).
    Eviction rate tells you if max_entries is too low.
    Promotion count tells you how much data graduated to Postgres.
    """

    hits: int = 0
    misses: int = 0
    admissions: int = 0
    evictions: int = 0
    expirations: int = 0
    promotions: int = 0

    @property
    def hit_rate(self) -> float:
        """Hit rate as a fraction [0.0, 1.0]. Returns 0 if no accesses."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "admissions": self.admissions,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "promotions": self.promotions,
            "hit_rate": round(self.hit_rate, 4),
        }
