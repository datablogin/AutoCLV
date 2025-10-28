"""Tests for Phase 5: Natural Language Interface

Tests for LLM-powered query parsing, result synthesis, conversational analysis,
and cost optimization features.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from analytics.services.mcp_server.orchestration.query_interpreter import (
    ParsedIntent,
    QueryInterpreter,
)
from analytics.services.mcp_server.orchestration.result_synthesizer import (
    ResultSynthesizer,
    SynthesizedResults,
)
from analytics.services.mcp_server.orchestration.query_cache import QueryCache
from analytics.services.mcp_server.orchestration.coordinator import (
    FourLensesCoordinator,
)


# ============================================================================
# Test QueryInterpreter (5.1)
# ============================================================================


class TestQueryInterpreter:
    """Test LLM-powered query interpretation."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic API client."""
        mock_client = AsyncMock()

        # Mock successful API response
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='```json\n{"lenses": ["lens1"], "date_range": null, '
                '"filters": {}, "parameters": {}, '
                '"reasoning": "Query mentions health, which maps to Lens 1"}\n```'
            )
        ]
        mock_response.usage = Mock(input_tokens=150, output_tokens=80)

        mock_client.messages.create = AsyncMock(return_value=mock_response)
        return mock_client

    @pytest.mark.asyncio
    async def test_query_interpreter_initialization(self):
        """Test QueryInterpreter initializes correctly."""
        interpreter = QueryInterpreter(api_key="test-key")

        assert interpreter.client is not None
        assert interpreter.model == "claude-3-5-sonnet-20241022"
        assert interpreter._total_input_tokens == 0
        assert interpreter._total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_parse_query_success(self, mock_anthropic_client):
        """Test successful query parsing with Claude."""
        with patch(
            "analytics.services.mcp_server.orchestration.query_interpreter.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            interpreter = QueryInterpreter(api_key="test-key")
            result = await interpreter.parse_query("Show me customer health")

        assert isinstance(result, ParsedIntent)
        assert result.lenses == ["lens1"]
        assert result.date_range is None
        assert result.filters == {}
        assert "health" in result.reasoning.lower()

    @pytest.mark.asyncio
    async def test_parse_query_tracks_tokens(self, mock_anthropic_client):
        """Test that query parsing tracks token usage."""
        with patch(
            "analytics.services.mcp_server.orchestration.query_interpreter.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            interpreter = QueryInterpreter(api_key="test-key")
            await interpreter.parse_query("Show me customer health")

        usage = interpreter.get_token_usage()
        assert usage["input_tokens"] == 150
        assert usage["output_tokens"] == 80
        assert usage["total_tokens"] == 230

    @pytest.mark.asyncio
    async def test_parse_query_with_multiple_lenses(self):
        """Test parsing query that should trigger multiple lenses."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text='{"lenses": ["lens1", "lens5"], "date_range": null, '
                '"filters": {}, "parameters": {}, '
                '"reasoning": "Query asks for both snapshot and overall health"}'
            )
        ]
        mock_response.usage = Mock(input_tokens=160, output_tokens=90)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch(
            "analytics.services.mcp_server.orchestration.query_interpreter.AsyncAnthropic",
            return_value=mock_client,
        ):
            interpreter = QueryInterpreter(api_key="test-key")
            result = await interpreter.parse_query(
                "Give me a health snapshot and overall customer base health"
            )

        assert len(result.lenses) == 2
        assert "lens1" in result.lenses
        assert "lens5" in result.lenses

    @pytest.mark.asyncio
    async def test_parse_query_with_invalid_json(self):
        """Test error handling for invalid JSON response."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is not valid JSON")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch(
            "analytics.services.mcp_server.orchestration.query_interpreter.AsyncAnthropic",
            return_value=mock_client,
        ):
            interpreter = QueryInterpreter(api_key="test-key")

            with pytest.raises(ValueError, match="Failed to parse query"):
                await interpreter.parse_query("test query")


# ============================================================================
# Test ResultSynthesizer (5.2)
# ============================================================================


class TestResultSynthesizer:
    """Test LLM-powered result synthesis."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic API client."""
        mock_client = AsyncMock()

        # Mock successful API response
        mock_response = Mock()
        mock_response.content = [
            Mock(
                text="```json\n{\n"
                '  "summary": "Customer base shows moderate health with some concentration risk.",\n'
                '  "insights": ["Health score is 72/100", "Top 10% contribute 65% of revenue"],\n'
                '  "recommendations": ["Focus on retention programs", "Diversify customer base"],\n'
                '  "narrative": "Detailed analysis shows..."\n'
                "}\n```"
            )
        ]
        mock_response.usage = Mock(input_tokens=800, output_tokens=200)

        mock_client.messages.create = AsyncMock(return_value=mock_response)
        return mock_client

    @pytest.mark.asyncio
    async def test_result_synthesizer_initialization(self):
        """Test ResultSynthesizer initializes correctly."""
        synthesizer = ResultSynthesizer(api_key="test-key")

        assert synthesizer.client is not None
        assert synthesizer.model == "claude-3-5-sonnet-20241022"
        assert synthesizer._total_input_tokens == 0
        assert synthesizer._total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_synthesize_results_success(self, mock_anthropic_client):
        """Test successful result synthesis with Claude."""
        with patch(
            "analytics.services.mcp_server.orchestration.result_synthesizer.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            synthesizer = ResultSynthesizer(api_key="test-key")

            lens_results = {
                "lens1": {
                    "total_customers": 1000,
                    "customer_health_score": 72.0,
                    "concentration_risk": "medium",
                }
            }

            result = await synthesizer.synthesize(
                "Show me customer health", lens_results
            )

        assert isinstance(result, SynthesizedResults)
        assert "health" in result.summary.lower()
        assert len(result.insights) >= 2
        assert len(result.recommendations) >= 2
        assert len(result.narrative) > 0

    @pytest.mark.asyncio
    async def test_synthesize_tracks_tokens(self, mock_anthropic_client):
        """Test that synthesis tracks token usage."""
        with patch(
            "analytics.services.mcp_server.orchestration.result_synthesizer.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            synthesizer = ResultSynthesizer(api_key="test-key")

            lens_results = {"lens1": {"total_customers": 1000}}
            await synthesizer.synthesize("test query", lens_results)

        usage = synthesizer.get_token_usage()
        assert usage["input_tokens"] == 800
        assert usage["output_tokens"] == 200
        assert usage["total_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_synthesize_with_no_results(self, mock_anthropic_client):
        """Test synthesis with no valid lens results."""
        with patch(
            "analytics.services.mcp_server.orchestration.result_synthesizer.AsyncAnthropic",
            return_value=mock_anthropic_client,
        ):
            synthesizer = ResultSynthesizer(api_key="test-key")

            # All results are None (failed lenses)
            lens_results = {"lens1": None, "lens2": None}

            result = await synthesizer.synthesize("test query", lens_results)

        # Should return minimal synthesis without calling API
        assert "No lens results available" in result.summary
        assert len(result.insights) > 0
        assert len(result.recommendations) > 0


# ============================================================================
# Test QueryCache (5.5)
# ============================================================================


class TestQueryCache:
    """Test query result caching for cost optimization."""

    def test_cache_initialization(self):
        """Test QueryCache initializes with correct defaults."""
        cache = QueryCache()

        assert cache.max_size == 100
        assert cache.ttl_seconds == 3600
        assert cache.get_hit_rate() == 0.0

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = QueryCache()

        result = cache.get("test query", True)

        assert result is None
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.0

    def test_cache_hit(self):
        """Test cache hit returns stored result."""
        cache = QueryCache()

        # Store result
        test_result = {"lenses_executed": ["lens1"], "insights": ["Test insight"]}
        cache.set("test query", True, test_result)

        # Retrieve result
        result = cache.get("test query", True)

        assert result is not None
        assert result["lenses_executed"] == ["lens1"]

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 1.0

    def test_cache_respects_use_llm_flag(self):
        """Test that cache keys differ based on use_llm flag."""
        cache = QueryCache()

        # Store with use_llm=True
        cache.set("test query", True, {"result": "llm"})

        # Try to retrieve with use_llm=False (should be cache miss)
        result = cache.get("test query", False)

        assert result is None

        # Try to retrieve with use_llm=True (should be cache hit)
        result = cache.get("test query", True)

        assert result is not None
        assert result["result"] == "llm"

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QueryCache(max_size=2)

        # Add 3 entries (should evict first one)
        cache.set("query1", True, {"id": 1})
        cache.set("query2", True, {"id": 2})
        cache.set("query3", True, {"id": 3})

        # First entry should be evicted
        result = cache.get("query1", True)
        assert result is None

        # Second and third should still be cached
        result2 = cache.get("query2", True)
        result3 = cache.get("query3", True)

        assert result2 is not None
        assert result3 is not None

        stats = cache.get_stats()
        assert stats["evictions"] == 1
        assert stats["size"] == 2

    def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        import time

        cache = QueryCache(ttl_seconds=1)

        # Store result
        cache.set("test query", True, {"data": "test"})

        # Immediate retrieval should work
        result = cache.get("test query", True)
        assert result is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        result = cache.get("test query", True)
        assert result is None

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = QueryCache()

        # Add multiple entries
        cache.set("query1", True, {"id": 1})
        cache.set("query2", True, {"id": 2})

        # Clear cache
        cache.clear()

        # All entries should be gone
        stats = cache.get_stats()
        assert stats["size"] == 0

        result1 = cache.get("query1", True)
        result2 = cache.get("query2", True)

        assert result1 is None
        assert result2 is None

    # Issue #122: Enhanced cache key normalization tests
    def test_cache_normalization_removes_punctuation(self):
        """Test that punctuation is removed for cache key matching (Issue #122)."""
        cache = QueryCache()

        # Store with punctuation
        test_result = {"data": "test"}
        cache.set("What is customer health?", True, test_result)

        # Retrieve without punctuation - should be cache hit
        result = cache.get("What is customer health", True)

        assert result is not None
        assert result["data"] == "test"

    def test_cache_normalization_removes_stopwords(self):
        """Test that stopwords are removed for better cache hits (Issue #122)."""
        cache = QueryCache()

        # Store with stopwords
        test_result = {"data": "health analysis"}
        cache.set("What is customer health?", True, test_result)

        # These should all match the same cache key (stopwords removed)
        queries = [
            "Show me customer health",
            "Tell me about customer health",
            "Give me customer health",
            "customer health",  # Minimal form
            "Display the customer health",
        ]

        for query in queries:
            result = cache.get(query, True)
            assert result is not None, f"Query '{query}' should be cache hit"
            assert result["data"] == "health analysis"

        # Verify high hit rate
        stats = cache.get_stats()
        # 1 miss (initial store) + 5 hits
        assert stats["hits"] == 5
        assert stats["misses"] == 0  # No misses on retrieval
        assert stats["hit_rate"] == 1.0

    def test_cache_normalization_preserves_key_terms(self):
        """Test that important domain terms are preserved (Issue #122)."""
        cache = QueryCache()

        # Store query with specific terms
        cache.set("customer retention analysis", True, {"type": "retention"})

        # Different phrasing should match
        result = cache.get("Show me customer retention analysis", True)
        assert result is not None
        assert result["type"] == "retention"

        # But different key terms should NOT match
        result = cache.get("customer health analysis", True)
        assert result is None  # Different key term: health vs retention

    def test_cache_normalization_handles_case_insensitivity(self):
        """Test that case differences don't affect cache hits (Issue #122)."""
        cache = QueryCache()

        test_result = {"data": "case test"}
        cache.set("Customer Health", True, test_result)

        # Different case variations should all hit
        queries = [
            "customer health",
            "CUSTOMER HEALTH",
            "CuStOmEr HeAlTh",
        ]

        for query in queries:
            result = cache.get(query, True)
            assert result is not None
            assert result["data"] == "case test"

    def test_cache_normalization_handles_extra_whitespace(self):
        """Test that extra whitespace is normalized (Issue #122)."""
        cache = QueryCache()

        test_result = {"data": "whitespace test"}
        cache.set("customer   health", True, test_result)

        # Extra whitespace should be normalized
        result = cache.get("customer health", True)
        assert result is not None

        result = cache.get("customer     health", True)
        assert result is not None

        result = cache.get("  customer health  ", True)
        assert result is not None

    def test_cache_normalization_example_scenarios(self):
        """Test real-world query variations from Issue #122."""
        cache = QueryCache()

        # Scenario 1: Customer health queries
        test_result = {"analysis": "health"}
        cache.set("What is customer health?", True, test_result)

        # All these should match
        health_queries = [
            "Show me customer health",
            "Tell me about customer health",
            "customer health",
            "What is the customer health?",
            "Give me customer health please",
        ]

        for query in health_queries:
            result = cache.get(query, True)
            assert result is not None, f"Query '{query}' should match"
            assert result["analysis"] == "health"

        # Scenario 2: Retention queries
        cache.clear()
        cache.set("What is our retention?", True, {"type": "retention"})

        retention_queries = [
            "Show me retention",
            "Tell me about retention",
            "What is retention",
            "retention",
        ]

        for query in retention_queries:
            result = cache.get(query, True)
            assert result is not None
            assert result["type"] == "retention"

    def test_cache_normalization_hit_rate_improvement(self):
        """Test that normalization significantly improves hit rates (Issue #122)."""
        cache = QueryCache()

        # Simulate typical user queries (with variations)
        test_result = {"lens": "lens1"}

        # Store base query
        cache.set("customer health", True, test_result)

        # Simulate 10 user queries with natural language variations
        query_variations = [
            "Show me customer health",
            "What is customer health?",
            "Tell me about customer health",
            "Display customer health",
            "Give me the customer health",
            "Can you show customer health",
            "customer health please",
            "Show the customer health",
            "What is the customer health",
            "customer health",
        ]

        hits = 0
        for query in query_variations:
            result = cache.get(query, True)
            if result is not None:
                hits += 1

        # All 10 should hit (100% hit rate after initial store)
        assert hits == 10, f"Expected 10 hits, got {hits}"

        stats = cache.get_stats()
        assert stats["hit_rate"] == 1.0  # Perfect hit rate

    def test_cache_normalization_doesnt_over_normalize(self):
        """Test that important distinctions are preserved (Issue #122)."""
        cache = QueryCache()

        # Store different analyses
        cache.set("customer retention", True, {"type": "retention"})
        cache.set("customer health", True, {"type": "health"})
        cache.set("customer revenue", True, {"type": "revenue"})

        # Each should remain distinct
        result1 = cache.get("Show me customer retention", True)
        assert result1["type"] == "retention"

        result2 = cache.get("Show me customer health", True)
        assert result2["type"] == "health"

        result3 = cache.get("Show me customer revenue", True)
        assert result3["type"] == "revenue"

        # Verify no false positives
        stats = cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 0


# ============================================================================
# Test Coordinator with LLM (5.3)
# ============================================================================


class TestCoordinatorWithLLM:
    """Test FourLensesCoordinator with LLM features enabled."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization_without_llm(self):
        """Test coordinator initializes correctly without LLM (Phase 3 mode)."""
        coordinator = FourLensesCoordinator(use_llm=False)

        assert coordinator.use_llm is False
        assert coordinator.query_interpreter is None
        assert coordinator.result_synthesizer is None

    @pytest.mark.asyncio
    async def test_coordinator_initialization_with_llm_no_api_key(self):
        """Test coordinator raises error if LLM requested without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key required"):
                FourLensesCoordinator(use_llm=True)

    @pytest.mark.asyncio
    async def test_coordinator_initialization_with_llm_and_api_key(self):
        """Test coordinator initializes correctly with LLM and API key."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            coordinator = FourLensesCoordinator(use_llm=True)

        assert coordinator.use_llm is True
        assert coordinator.query_interpreter is not None
        assert coordinator.result_synthesizer is not None


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase5Integration:
    """Integration tests for Phase 5 features."""

    @pytest.mark.asyncio
    async def test_end_to_end_caching(self):
        """Test that caching reduces API calls on repeated queries."""
        cache = QueryCache()

        # Simulate two identical queries
        query = "Show me customer health"

        # First call (cache miss)
        result1 = cache.get(query, True)
        assert result1 is None

        # Simulate storing result
        test_result = {"lenses_executed": ["lens1"]}
        cache.set(query, True, test_result)

        # Second call (cache hit)
        result2 = cache.get(query, True)
        assert result2 is not None
        assert result2["lenses_executed"] == ["lens1"]

        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.5  # 1 hit, 1 miss = 50%

    def test_token_usage_tracking(self):
        """Test that token usage is tracked correctly."""
        interpreter = QueryInterpreter(api_key="test-key")
        synthesizer = ResultSynthesizer(api_key="test-key")

        # Initial state
        assert interpreter.get_token_usage()["total_tokens"] == 0
        assert synthesizer.get_token_usage()["total_tokens"] == 0

        # Reset should work
        interpreter.reset_token_usage()
        synthesizer.reset_token_usage()

        assert interpreter.get_token_usage()["total_tokens"] == 0
        assert synthesizer.get_token_usage()["total_tokens"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
