# Comprehensive Test: Phase 4A + Phase 5

Run this complete test sequence to validate all observability and LLM features.

---

## 🔍 PART 1: Phase 4A - Observability & Resilience

### Test 1.1: Initial Health Check
```
Run health_check

Expected: Server should be healthy, foundation data should show as "not available"
```

### Test 1.2: Load Sample Data
```
Run load_transactions with:
- path: tests/fixtures/synthetic_transactions_2023_2024.csv

Expected: Should load ~110,000 transactions for ~4,000 customers
```

### Test 1.3: Build Foundation
```
Run these in sequence:
1. build_customer_data_mart
2. calculate_rfm_metrics
3. create_customer_cohorts

Expected: All should succeed, creating data mart, RFM scores, and cohorts
```

### Test 1.4: Health Check After Data Load
```
Run health_check again

Expected: Foundation data should now show as "fully available"
```

### Test 1.5: Run Orchestrated Analyses
```
Run three analyses to generate metrics:

Analysis A:
run_orchestrated_analysis with:
- query: "Give me the overall customer base health"
- use_llm: false

Analysis B:
run_orchestrated_analysis with:
- query: "customer health snapshot"
- use_llm: false

Analysis C:
run_orchestrated_analysis with:
- query: "customer health and overall base health"
- use_llm: false

Expected: Each should execute appropriate lenses and track metrics
```

### Test 1.6: Check Execution Metrics
```
Run get_execution_metrics

Expected to see:
- Total analyses: 3
- Success rate: 100%
- Per-lens statistics (lens1 and lens5 should have executions)
- Average duration metrics
```

### Test 1.7: Final Health Check
```
Run health_check one more time

Expected: Should show updated execution counts and system uptime
```

---

## 🤖 PART 2: Phase 5 - Natural Language Interface

### Test 2.1: Rule-Based Baseline (No LLM)
```
Run run_orchestrated_analysis with:
- query: "customer health snapshot"
- use_llm: false
- use_cache: false

Expected:
- Should execute Lens 1
- Simple bullet point insights
- No narrative field
- Fast execution (<100ms)
```

### Test 2.2: First LLM Query (Cold - Cache Miss)
```
Run run_orchestrated_analysis with:
- query: "Tell me about customer health and which cohorts are performing best"
- use_llm: true
- use_cache: true

Expected:
- Should take 2-5 seconds
- Should execute multiple lenses (Lens 1 + Lens 4 or Lens 5)
- Should include narrative field with detailed explanation
- cache_hit: false
- token_usage should show ~700-1400 total tokens
- Cost: ~$0.05-0.10
```

### Test 2.3: Cached LLM Query (Warm - Cache Hit)
```
Run EXACT same query as 2.2:
run_orchestrated_analysis with:
- query: "Tell me about customer health and which cohorts are performing best"
- use_llm: true
- use_cache: true

Expected:
- Should be nearly instant (<100ms)
- cache_hit: true
- cache_stats.hit_rate: 0.5 (50%)
- Identical results to 2.2
- Cost: $0.00 (no API call)
```

### Test 2.4: Different LLM Query
```
Run run_orchestrated_analysis with:
- query: "What's the retention trend and overall customer base health?"
- use_llm: true
- use_cache: true

Expected:
- cache_hit: false (different query)
- Should identify Lens 2 and Lens 5
- New narrative with different focus
- cache_stats.size: 2
```

### Test 2.5: Conversational Analysis (Turn 1)
```
Run run_conversational_analysis with:
- query: "Give me a comprehensive customer base analysis"
- use_llm: true
- conversation_history: null

Expected:
- conversation_turn: 1
- Should execute multiple lenses
- conversation_history array with one entry
- token_usage tracked
```

### Test 2.6: Conversational Analysis (Turn 2)
```
Run run_conversational_analysis with:
- query: "Now focus on which cohorts need attention"
- use_llm: true
- conversation_history: [paste the conversation_history from 2.5 result]

Expected:
- conversation_turn: 2
- Should understand context from Turn 1
- conversation_history array with two entries
- Cumulative token_usage
```

### Test 2.7: Check Final Metrics and Cache
```
Run get_execution_metrics again

Expected to see:
- Total analyses increased (now 3 + 6 = 9 analyses)
- More lens executions recorded
- Success rate still 100%
```

---

## 📊 RESULTS TO REPORT

### Phase 4A Validation Checklist

- [ ] Initial health check shows "not available" for foundation data
- [ ] After loading data, health check shows "fully available"
- [ ] All 3 orchestrated analyses succeeded
- [ ] Execution metrics show:
  - Total analyses: 3 (at least)
  - Success rate: 100%
  - Per-lens stats populated
- [ ] Final health check shows updated metrics

### Phase 5 Validation Checklist

- [ ] Rule-based mode works without API key
- [ ] LLM mode (Test 2.2) generates narrative
- [ ] Cache hit (Test 2.3) is instant and returns same results
- [ ] cache_stats.hit_rate increases correctly
- [ ] Token usage is tracked and reasonable (<1500 tokens/query)
- [ ] Conversational analysis maintains context across turns
- [ ] No errors or crashes

### Performance Benchmarks

| Test | Expected | Actual |
|------|----------|--------|
| Test 2.1 (Rule-based) | <100ms | ___ms |
| Test 2.2 (LLM cold) | 2-5s | ___ms |
| Test 2.3 (LLM cached) | <100ms | ___ms |
| Test 2.4 (LLM new) | 2-5s | ___ms |

### Cost Analysis

| Test | Expected Cost | Actual |
|------|---------------|--------|
| Test 2.2 | ~$0.05-0.10 | $____ |
| Test 2.3 | $0.00 (cached) | $____ |
| Test 2.4 | ~$0.05-0.10 | $____ |
| Test 2.5 | ~$0.05-0.10 | $____ |
| Test 2.6 | ~$0.05-0.10 | $____ |
| **Total** | ~$0.25-0.50 | $____ |

### Narrative Quality Assessment

Compare the outputs:

**Test 2.1 (Rule-based insights):**
- Format: Simple bullet points
- Depth: Basic metrics
- Actionability: Generic recommendations

**Test 2.2 (LLM narrative):**
- Format: Cohesive narrative + bullet points
- Depth: Business context and interpretation
- Actionability: Specific, prioritized recommendations

Rate the narrative quality (1-5):
- Coherence: ___/5
- Actionability: ___/5
- Business value: ___/5

---

## 🐛 Issues to Report

If any tests fail or produce unexpected results, note:

1. **Test number:**
2. **Error message:**
3. **Expected vs actual behavior:**
4. **Screenshots (if applicable):**

---

## ✅ Success Criteria

**Both phases are successful if:**

Phase 4A:
- ✅ Health checks work correctly
- ✅ Execution metrics are tracked
- ✅ All analyses succeed
- ✅ Metrics show accurate counts

Phase 5:
- ✅ Rule-based mode works (backward compatible)
- ✅ LLM mode generates narratives
- ✅ Caching reduces latency (instant on repeat)
- ✅ Token usage is reasonable (<$0.10/query)
- ✅ Conversational context is maintained
- ✅ No API errors or crashes

---

## 🎯 Final Validation

After completing all tests:

1. **Total analyses run:** ___
2. **Overall success rate:** ___%
3. **Cache hit rate:** ___%
4. **Total cost:** $____
5. **Any issues found:** Yes / No
6. **Phase 4A status:** PASS / FAIL
7. **Phase 5 status:** PASS / FAIL

**Overall Assessment:** ☐ PASS ☐ FAIL ☐ PARTIAL

**Tester signature:** ___________
**Date:** ___________
