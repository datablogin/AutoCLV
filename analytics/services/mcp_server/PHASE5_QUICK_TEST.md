# Phase 5 Quick Testing Prompts

Copy and paste these prompts directly into Claude Desktop to test Phase 5 features.

---

## Setup (Do This First)

**1. Load Sample Data**

```
Please use the load_transactions tool to load sample transaction data from:
data/sample_transactions.csv

Then use build_customer_data_mart to build the data mart.
Then use calculate_rfm_metrics to calculate RFM metrics.
Then use create_customer_cohorts to create quarterly cohorts.
```

---

## Test Suite 1: Baseline (No LLM)

**Test 1A: Simple Query - Rule-Based**

```
Please use run_orchestrated_analysis with these parameters:
- query: "customer health snapshot"
- use_llm: false
- use_cache: false

This should execute Lens 1 using rule-based keyword matching.
```

**Test 1B: Multi-Lens Query - Rule-Based**

```
Please use run_orchestrated_analysis with:
- query: "overall customer base health and cohorts"
- use_llm: false
- use_cache: false

This should execute multiple lenses based on keywords.
```

---

## Test Suite 2: LLM-Powered Parsing

**Test 2A: Ambiguous Natural Language**

```
Please use run_orchestrated_analysis with:
- query: "Tell me about my customers and how they're doing"
- use_llm: true
- use_cache: false

Claude should interpret this and select appropriate lenses, then generate a narrative explanation.
```

**Test 2B: Complex Multi-Aspect Query**

```
Please use run_orchestrated_analysis with:
- query: "I need to understand retention trends, identify which customer cohorts are performing best, and get an overall assessment of customer base health"
- use_llm: true
- use_cache: false

Claude should identify that this requires Lens 2 (retention), Lens 4 (cohorts), and Lens 5 (overall health), then synthesize results into a coherent narrative.
```

**Test 2C: Business-Focused Query**

```
Please use run_orchestrated_analysis with:
- query: "Are we retaining customers well? Which acquisition cohorts should we focus on?"
- use_llm: true
- use_cache: false

Claude should interpret business intent and provide actionable insights.
```

---

## Test Suite 3: Query Caching

**Test 3A: Demonstrate Cache Miss Then Hit**

```
First, run this query:
Please use run_orchestrated_analysis with:
- query: "Show me customer health metrics"
- use_llm: true
- use_cache: true

Then immediately run the EXACT SAME query again:
Please use run_orchestrated_analysis with:
- query: "Show me customer health metrics"
- use_llm: true
- use_cache: true

The first call should be a cache miss (cache_hit: false) and take 2-5 seconds.
The second call should be a cache hit (cache_hit: true) and return instantly.
Check the cache_stats to see hit_rate increase to 0.5 (50%).
```

**Test 3B: Different Queries Build Cache**

```
Run these 3 different queries in sequence:
1. query: "customer health snapshot", use_llm: true, use_cache: true
2. query: "cohort analysis", use_llm: true, use_cache: true
3. query: "overall base health", use_llm: true, use_cache: true

Then repeat query #1 - it should be cached!

Check cache_stats.size - should be 3.
```

---

## Test Suite 4: Conversational Analysis

**Test 4A: Multi-Turn Conversation**

```
Turn 1: Please use run_conversational_analysis with:
- query: "Give me a snapshot of current customer base health"
- use_llm: true
- conversation_history: null

After you get the result, continue with:

Turn 2: Please use run_conversational_analysis with:
- query: "Now show me which cohorts are performing best"
- use_llm: true
- conversation_history: [paste the conversation_history from Turn 1 result]

After you get that result:

Turn 3: Please use run_conversational_analysis with:
- query: "What actions should we take based on these findings?"
- use_llm: true
- conversation_history: [paste the updated conversation_history from Turn 2]

This demonstrates context maintenance across multiple queries. Check that conversation_turn increments and token_usage is tracked.
```

---

## Test Suite 5: Narrative Quality

**Test 5A: Single Lens Narrative**

```
Please use run_orchestrated_analysis with:
- query: "Analyze current customer base health"
- use_llm: true
- use_cache: false

Examine the response for:
- summary: Should be 2-3 sentences capturing key findings
- insights: Should be 3-5 actionable bullet points
- recommendations: Should be specific next steps
- narrative: Should be a detailed explanation connecting metrics to business impact
```

**Test 5B: Multi-Lens Synthesis**

```
Please use run_orchestrated_analysis with:
- query: "Give me a comprehensive analysis covering customer health, cohort performance, and retention trends"
- use_llm: true
- use_cache: false

Claude should execute multiple lenses and synthesize results into a coherent narrative that:
- Connects insights across lenses
- Prioritizes most important findings
- Provides business context
- Avoids excessive technical jargon
```

---

## Test Suite 6: Error Handling

**Test 6A: Graceful Fallback (Simulate Missing API Key)**

```
Please use run_orchestrated_analysis with:
- query: "customer health"
- use_llm: false
- use_cache: false

This should work fine without requiring an API key (rule-based mode).
```

**Test 6B: Empty Query Handling**

```
Please use run_orchestrated_analysis with:
- query: "health"
- use_llm: true
- use_cache: false

Should default to Lens 1 (health snapshot).
```

---

## Test Suite 7: Cost Analysis

**Test 7A: Token Usage Tracking**

```
Please use run_conversational_analysis with:
- query: "Comprehensive analysis of customer base with all available lenses"
- use_llm: true
- conversation_history: null

After completion, examine token_usage:
- query_parsing_input/output: Tokens used for intent parsing
- synthesis_input/output: Tokens used for result synthesis
- total_tokens: Overall token count

Verify total cost is under $0.50 (should be ~$0.05-0.10).
```

---

## Success Indicators

After running all tests, you should observe:

✅ **Rule-based mode** works without API key (Tests 1A, 1B)
✅ **LLM parsing** correctly interprets natural language (Tests 2A, 2B, 2C)
✅ **Caching** reduces latency on repeated queries (Tests 3A, 3B)
✅ **Conversation** maintains context across turns (Test 4A)
✅ **Narratives** are coherent and actionable (Tests 5A, 5B)
✅ **Error handling** is graceful (Tests 6A, 6B)
✅ **Costs** are under target (Test 7A)

---

## Performance Expectations

| Test | Expected Latency | Expected Cost |
|------|------------------|---------------|
| Rule-based (1A, 1B) | <500ms | $0.00 |
| LLM cold (2A, 2B, 2C) | 2-5s | $0.05-0.10 |
| LLM cached (3A repeat) | <500ms | $0.00 |
| Conversation (4A, 3 turns) | 6-15s total | $0.15-0.30 |

---

## Quick Visual Test

**The Simplest Test** - Copy/Paste This:

```
Please perform this complete test sequence:

1. Load sample data:
   - load_transactions: data/sample_transactions.csv
   - build_customer_data_mart
   - calculate_rfm_metrics
   - create_customer_cohorts

2. Test rule-based mode:
   run_orchestrated_analysis with query="customer health", use_llm=false

3. Test LLM mode with caching:
   a) run_orchestrated_analysis with query="Tell me about customer health and which cohorts perform best", use_llm=true, use_cache=true
   b) Repeat exact same query - should be instant (cached)

4. Show me:
   - Both results side-by-side
   - Cache statistics
   - Token usage from LLM mode
   - Whether the narrative in LLM mode is more actionable than rule-based insights

This demonstrates all Phase 5 features in one go!
```

---

## Troubleshooting

**If you get "ANTHROPIC_API_KEY required"**:
- Check Claude Desktop config includes ANTHROPIC_API_KEY
- Restart Claude Desktop after config changes
- Use use_llm=false for testing without API key

**If queries are slow**:
- First query with LLM is 2-5s (normal)
- Subsequent identical queries should be instant (cached)
- Check use_cache=true

**If narrative is missing**:
- Only present when use_llm=true
- Check that API key is valid
- Look for errors in response

**If caching isn't working**:
- Queries must be EXACTLY identical (case-sensitive)
- use_cache must be true
- Check cache_stats in response
