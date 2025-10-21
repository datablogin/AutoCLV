# Phase 5 Testing Guide: Natural Language Interface

This guide provides comprehensive testing scenarios for Phase 5 features including LLM-powered query parsing, result synthesis, conversational analysis, and query caching.

## Prerequisites

### 1. Install Dependencies
```bash
uv sync  # Install anthropic package
```

### 2. Configure Claude Desktop

Update your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "four-lenses-analytics": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a",
        "run",
        "fastmcp",
        "run",
        "analytics/services/mcp_server/main.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

**Important**: Replace with your actual:
- Project path
- Anthropic API key (get from https://console.anthropic.com/)

### 3. Prepare Test Data

Load sample transaction data first:

```
Use the load_transactions tool with the sample data at:
data/sample_transactions.csv
```

---

## Test Scenarios

### Scenario 1: Baseline Rule-Based Mode (Phase 3)

**Objective**: Verify backward compatibility - Phase 5 should work without LLM

**Test 1.1**: Simple query without LLM
```
Use run_orchestrated_analysis with:
- query: "customer health snapshot"
- use_llm: false
- use_cache: false
```

**Expected Result**:
- ✅ Should execute Lens 1 (rule-based keyword matching)
- ✅ Should return insights without narrative
- ✅ cache_hit should be false
- ✅ No API key errors

**Test 1.2**: Multi-lens query without LLM
```
Use run_orchestrated_analysis with:
- query: "overall customer base health and cohorts"
- use_llm: false
- use_cache: false
```

**Expected Result**:
- ✅ Should execute Lens 5 and Lens 4 (keyword: "overall" + "cohorts")
- ✅ Should aggregate results without LLM synthesis
- ✅ Fast response (<1s)

---

### Scenario 2: LLM-Powered Query Parsing

**Objective**: Verify Claude correctly interprets natural language queries

**Test 2.1**: Ambiguous query
```
Use run_orchestrated_analysis with:
- query: "Tell me about my customers"
- use_llm: true
- use_cache: false
```

**Expected Result**:
- ✅ Claude should parse intent and select appropriate lens (likely Lens 1)
- ✅ Should include reasoning in logs
- ✅ Should return narrative field with detailed explanation
- ✅ Execution time 2-5s

**Test 2.2**: Complex multi-aspect query
```
Use run_orchestrated_analysis with:
- query: "I want to understand customer retention, see which cohorts perform best, and get an overall health assessment"
- use_llm: true
- use_cache: false
```

**Expected Result**:
- ✅ Should identify multiple lenses needed (Lens 2, Lens 4, Lens 5)
- ✅ Should execute lenses in parallel where possible
- ✅ Should generate coherent narrative combining all lens results
- ✅ Summary should mention retention, cohort performance, and overall health

**Test 2.3**: Date-based query
```
Use run_orchestrated_analysis with:
- query: "Compare Q1 2024 to Q4 2023"
- use_llm: true
- use_cache: false
```

**Expected Result**:
- ✅ Should identify Lens 2 (period comparison)
- ✅ Should extract date ranges from query
- ✅ Reasoning should mention "comparison" keyword

---

### Scenario 3: LLM Result Synthesis

**Objective**: Verify narrative generation quality

**Test 3.1**: Single lens synthesis
```
Use run_orchestrated_analysis with:
- query: "customer health snapshot"
- use_llm: true
- use_cache: false
```

**Expected Result**:
- ✅ Should return narrative field
- ✅ Summary should be 2-3 sentences
- ✅ Insights should be 3-5 actionable bullet points
- ✅ Recommendations should be specific and actionable
- ✅ Narrative should explain metrics in business terms

**Test 3.2**: Multi-lens synthesis
```
Use run_orchestrated_analysis with:
- query: "overall base health and cohort analysis"
- use_llm: true
- use_cache: false
```

**Expected Result**:
- ✅ Narrative should connect insights across Lens 5 and Lens 4
- ✅ Should highlight patterns across lenses
- ✅ Should prioritize most important findings
- ✅ Should avoid technical jargon where possible

---

### Scenario 4: Query Caching

**Objective**: Verify cost optimization through caching

**Test 4.1**: Cache miss then hit
```
First call:
Use run_orchestrated_analysis with:
- query: "Show me customer health"
- use_llm: true
- use_cache: true

Second call (identical query):
Use run_orchestrated_analysis with:
- query: "Show me customer health"
- use_llm: true
- use_cache: true
```

**Expected Results**:
- First call:
  - ✅ cache_hit: false
  - ✅ Execution time: 2-5s
  - ✅ cache_stats.misses: 1

- Second call:
  - ✅ cache_hit: true
  - ✅ Execution time: <500ms (instant)
  - ✅ cache_stats.hit_rate: 0.5 (50%)
  - ✅ Same results as first call

**Test 4.2**: Cache key respects use_llm flag
```
Call 1:
- query: "customer health"
- use_llm: true

Call 2:
- query: "customer health"
- use_llm: false
```

**Expected Result**:
- ✅ Both should be cache misses (different cache keys)
- ✅ Results may differ (LLM has narrative, non-LLM doesn't)

**Test 4.3**: Cache statistics tracking
```
After multiple queries, check cache_stats:
```

**Expected Result**:
- ✅ hit_rate between 0.0 and 1.0
- ✅ size <= max_size (100)
- ✅ hits + misses = total queries

---

### Scenario 5: Conversational Analysis

**Objective**: Verify conversation context maintenance

**Test 5.1**: Multi-turn conversation
```
Turn 1:
Use run_conversational_analysis with:
- query: "Give me a customer health snapshot"
- use_llm: true
- conversation_history: null

Turn 2:
Use run_conversational_analysis with:
- query: "Now show me cohort analysis"
- use_llm: true
- conversation_history: [result from Turn 1]

Turn 3:
Use run_conversational_analysis with:
- query: "Which cohort performs best?"
- use_llm: true
- conversation_history: [results from Turn 1 and 2]
```

**Expected Results**:
- Turn 1:
  - ✅ conversation_turn: 1
  - ✅ Should execute Lens 1

- Turn 2:
  - ✅ conversation_turn: 2
  - ✅ Should execute Lens 4 (cohort analysis)
  - ✅ conversation_history should contain both queries

- Turn 3:
  - ✅ conversation_turn: 3
  - ✅ Should identify specific cohort from previous results
  - ✅ Full conversation history maintained

**Test 5.2**: Token usage tracking
```
After Turn 3, check token_usage field:
```

**Expected Result**:
- ✅ query_parsing_input: >0
- ✅ query_parsing_output: >0
- ✅ synthesis_input: >0
- ✅ synthesis_output: >0
- ✅ total_tokens: sum of all tokens
- ✅ Cumulative across conversation turns

---

### Scenario 6: Error Handling & Fallbacks

**Objective**: Verify graceful degradation

**Test 6.1**: Missing API key
```
Remove ANTHROPIC_API_KEY from environment

Use run_orchestrated_analysis with:
- query: "customer health"
- use_llm: true
```

**Expected Result**:
- ✅ Should return error explaining API key required
- ✅ Should NOT crash
- ✅ Should suggest using use_llm: false

**Test 6.2**: Invalid query
```
Use run_orchestrated_analysis with:
- query: ""
- use_llm: true
```

**Expected Result**:
- ✅ Should handle gracefully
- ✅ Should default to Lens 1 or return helpful error

**Test 6.3**: Partial lens failures
```
Use run_orchestrated_analysis on data with missing cohorts:
- query: "overall health and cohort analysis"
- use_llm: true
```

**Expected Result**:
- ✅ Should execute successfully executing lenses (Lens 5 may work, Lens 4 may fail)
- ✅ lenses_failed should list failed lenses
- ✅ Should still synthesize results from successful lenses
- ✅ Narrative should acknowledge partial results

---

### Scenario 7: Cost Monitoring

**Objective**: Verify cost tracking and optimization

**Test 7.1**: Track token usage
```
Run 5 different queries with use_llm: true and use_cache: false

For each, note the token_usage in cache_stats
```

**Expected Results**:
- ✅ Query parsing: ~150-200 input tokens, ~50-100 output tokens
- ✅ Synthesis: ~500-1000 input tokens, ~200-400 output tokens
- ✅ Total per query: ~700-1400 tokens
- ✅ Cost per query: ~$0.05-0.10

**Test 7.2**: Cache effectiveness
```
Run same query 10 times:
5 times with different queries (cache misses)
5 times repeating queries (cache hits)

Check cache_stats.hit_rate
```

**Expected Result**:
- ✅ hit_rate should be ~0.5 (50%)
- ✅ Cost reduction: ~50% compared to no caching

---

## Success Criteria

### Phase 5 is successful if:

- [x] Rule-based mode works without API key (backward compatibility)
- [ ] LLM mode correctly interprets natural language queries
- [ ] LLM mode generates coherent, actionable narratives
- [ ] Caching reduces costs on repeated queries
- [ ] Conversational analysis maintains context across turns
- [ ] Token usage is tracked and within budget (<$0.50 per query)
- [ ] Graceful fallback to rules when LLM fails
- [ ] No regressions in existing functionality (456 tests pass)

---

## Performance Benchmarks

Expected latency targets:

| Mode | Cache Status | Target Latency | Measured |
|------|--------------|----------------|----------|
| Rule-based | N/A | <500ms | ___ms |
| LLM | Cold (miss) | <5s | ___ms |
| LLM | Warm (hit) | <500ms | ___ms |

Expected cost targets:

| Scenario | Target Cost | Measured |
|----------|-------------|----------|
| Cold query (no cache) | <$0.10 | $____ |
| Cached query | $0.00 | $____ |
| 10 queries (50% hit rate) | <$0.50 | $____ |

---

## Debugging Tips

### Enable debug logging:
Add to Claude Desktop config:
```json
"env": {
  "LOG_LEVEL": "DEBUG"
}
```

### Check MCP server logs:
```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

### Test individual components:
```bash
# Test query interpreter
python -m pytest tests/services/mcp_server/test_phase5_natural_language.py::TestQueryInterpreter -v

# Test result synthesizer
python -m pytest tests/services/mcp_server/test_phase5_natural_language.py::TestResultSynthesizer -v

# Test caching
python -m pytest tests/services/mcp_server/test_phase5_natural_language.py::TestQueryCache -v
```

---

## Common Issues

### Issue: "No module named 'anthropic'"
**Solution**: Run `uv sync` or `pip install anthropic`

### Issue: "Anthropic API key required"
**Solution**: Add ANTHROPIC_API_KEY to Claude Desktop config

### Issue: Cache not working
**Solution**: Ensure use_cache: true and queries are identical (case-sensitive)

### Issue: Slow LLM responses
**Solution**:
- Check internet connection
- Verify API key is valid
- Try with use_cache: true for repeated queries

---

## Next Steps After Testing

1. Document any issues found in GitHub Issue #98
2. Gather user feedback on narrative quality
3. Tune prompts if needed for better intent parsing
4. Adjust cache size/TTL based on usage patterns
5. Monitor token usage in production
6. Consider implementing Phase 4B for advanced observability

---

## Testing Checklist

- [ ] Scenario 1: Baseline rule-based mode
- [ ] Scenario 2: LLM query parsing
- [ ] Scenario 3: LLM result synthesis
- [ ] Scenario 4: Query caching
- [ ] Scenario 5: Conversational analysis
- [ ] Scenario 6: Error handling
- [ ] Scenario 7: Cost monitoring
- [ ] Performance benchmarks recorded
- [ ] All success criteria met
- [ ] Issues documented (if any)

**Testing Date**: _________
**Tester**: _________
**API Key Used**: _________ (last 4 chars)
**Overall Result**: ☐ PASS ☐ FAIL ☐ PARTIAL
