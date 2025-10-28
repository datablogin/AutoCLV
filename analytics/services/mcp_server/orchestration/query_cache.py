"""Query Cache - Phase 5 Cost Optimization

This module provides a simple in-memory cache for query results to reduce
LLM API costs when users repeat similar queries.

Design:
- LRU cache with configurable max size and TTL
- Thread-safe for concurrent access
- Tracks cache hit/miss rates for monitoring
- Can be disabled via environment variable
- Enhanced query normalization for better cache hit rates (Issue #122)

Query Normalization Strategy:
- Remove punctuation (? ! . , etc.)
- Remove common stopwords (show, tell, me, about, the, etc.)
- Normalize whitespace (multiple spaces → single space)
- Lowercase all text
- Target: Increase hit rate from ~30% to 50%+
"""

import hashlib
import re
import threading
import time
from collections import OrderedDict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Common English stopwords to remove for better cache hit rates
# Focus on words that don't change query intent (interrogatives, articles, etc.)
_STOPWORDS = {
    # Interrogatives and request words
    "show",
    "tell",
    "give",
    "display",
    "what",
    "how",
    "when",
    "where",
    "why",
    "can",
    "could",
    "would",
    "should",
    # Articles and determiners
    "a",
    "an",
    "the",
    # Common prepositions and pronouns
    "me",
    "my",
    "you",
    "your",
    "our",
    "about",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    # Polite words
    "please",
    "thanks",
    "thank",
    # Common conjunctions
    "and",
    "or",
    "but",
    # Common verbs that don't affect intent
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "get",
    "got",
}


class QueryCache:
    """LRU cache for query results with TTL support.

    This cache reduces LLM API costs by storing recent query results.
    Cache keys are based on query hash + use_llm flag.

    Cost impact:
    - Cold query (cache miss): Full LLM cost (~$0.05-0.10)
    - Warm query (cache hit): No LLM cost ($0.00)
    - Expected cost reduction: 30-50% with typical usage patterns
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """Initialize query cache.

        Args:
            max_size: Maximum number of cached queries (default: 100)
            ttl_seconds: Time-to-live for cache entries in seconds (default: 3600 = 1 hour)
        """
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        logger.info(
            "query_cache_initialized",
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )

    def get(self, query: str, use_llm: bool) -> dict[str, Any] | None:
        """Get cached result for query.

        Args:
            query: User query string
            use_llm: Whether LLM is being used (affects cache key)

        Returns:
            Cached result dict or None if not found/expired
        """
        cache_key = self._make_key(query, use_llm)

        with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                logger.debug("cache_miss", query_hash=cache_key[:16])
                return None

            entry = self._cache[cache_key]

            # Check TTL
            age = time.time() - entry["timestamp"]
            if age > self.ttl_seconds:
                # Entry expired - use try/except to handle race condition
                try:
                    del self._cache[cache_key]
                except KeyError:
                    pass  # Already deleted by another thread
                self._misses += 1
                logger.debug(
                    "cache_expired",
                    query_hash=cache_key[:16],
                    age_seconds=int(age),
                )
                return None

            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(cache_key)
            self._hits += 1

            logger.info(
                "cache_hit",
                query_hash=cache_key[:16],
                age_seconds=int(age),
                hit_rate=self.get_hit_rate(),
            )

            return entry["result"]

    def set(self, query: str, use_llm: bool, result: dict[str, Any]) -> None:
        """Store query result in cache.

        Args:
            query: User query string
            use_llm: Whether LLM was used (affects cache key)
            result: Analysis result to cache
        """
        cache_key = self._make_key(query, use_llm)

        with self._lock:
            # Add/update entry
            self._cache[cache_key] = {
                "result": result,
                "timestamp": time.time(),
            }

            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)

            # Evict oldest entry if cache is full
            if len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
                logger.debug("cache_eviction", total_evictions=self._evictions)

            logger.debug(
                "cache_set",
                query_hash=cache_key[:16],
                cache_size=len(self._cache),
            )

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            entries_cleared = len(self._cache)
            self._cache.clear()
            logger.info("cache_cleared", entries_cleared=entries_cleared)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, size, evictions
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self.max_size,
                "evictions": self._evictions,
                "ttl_seconds": self.ttl_seconds,
            }

    def get_hit_rate(self) -> float:
        """Get cache hit rate (0.0 to 1.0).

        Returns:
            Hit rate as fraction (e.g., 0.75 = 75% hit rate)
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def _make_key(self, query: str, use_llm: bool) -> str:
        """Create cache key from query and use_llm flag.

        Enhanced normalization (Issue #122) to improve cache hit rates:
        1. Lowercase
        2. Remove punctuation
        3. Remove stopwords (show, tell, me, about, etc.)
        4. Normalize whitespace

        Examples of queries that now map to same key:
        - "What is customer health?" → "customer health"
        - "Show me customer health" → "customer health"
        - "Tell me about customer health" → "customer health"

        Uses SHA-256 hash for consistent key generation.

        Args:
            query: User query string
            use_llm: Whether LLM is being used

        Returns:
            Cache key (64-character hex string)
        """
        # Step 1: Lowercase
        normalized = query.lower().strip()

        # Step 2: Remove punctuation (but keep spaces and alphanumerics)
        normalized = re.sub(r"[^\w\s]", "", normalized)

        # Step 3: Remove stopwords
        words = normalized.split()
        filtered_words = [w for w in words if w not in _STOPWORDS]

        # Step 4: Normalize whitespace (rejoin with single spaces)
        normalized = " ".join(filtered_words)

        # Include use_llm in key (results differ between LLM and rule-based)
        key_input = f"{normalized}|{use_llm}"

        return hashlib.sha256(key_input.encode()).hexdigest()


# Global cache instance (singleton) with thread-safe initialization
_global_cache: QueryCache | None = None
_cache_lock = threading.Lock()


def get_query_cache() -> QueryCache:
    """Get global query cache instance with thread-safe initialization.

    Uses double-checked locking pattern to ensure thread-safe singleton initialization.

    Returns:
        Global QueryCache singleton
    """
    global _global_cache
    if _global_cache is None:
        with _cache_lock:
            # Double-check locking pattern
            if _global_cache is None:
                _global_cache = QueryCache()
    return _global_cache
