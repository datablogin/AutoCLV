# Phase 5: Natural Language Interface - Setup & Testing

Version 2.0.0 - Completed October 2024

## What's New in Phase 5

Phase 5 adds **optional LLM-powered features** to the Four Lenses Analytics MCP server:

- 🤖 **Claude-powered query parsing**: Natural language → structured intent
- 📝 **Narrative synthesis**: Multi-lens results → coherent stories
- 💬 **Conversational analysis**: Follow-up questions with context
- 💰 **Query caching**: 30-50% cost reduction on repeated queries
- 📊 **Token monitoring**: Real-time cost tracking

**Key Design**: Phase 5 is **100% backward compatible**. All features work without an API key using rule-based mode (Phase 3).

---

## Quick Start

### 1. Update Claude Desktop Config

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

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
        "ANTHROPIC_API_KEY": "sk-ant-api03-..."
      }
    }
  }
}
```

**Important**:
- Replace path with your actual project path
- Get API key from: https://console.anthropic.com/
- **Restart Claude Desktop** after config changes

### 2. Install Dependencies

```bash
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
uv sync  # Installs anthropic package
```

### 3. Test It

Open Claude Desktop and paste this:

```
Please test Phase 5 by running these commands:

1. Load sample data:
   load_transactions with file: data/sample_transactions.csv

2. Build foundation:
   - build_customer_data_mart
   - calculate_rfm_metrics
   - create_customer_cohorts

3. Test LLM features:
   run_orchestrated_analysis with:
   - query: "Tell me about customer health and which cohorts perform best"
   - use_llm: true
   - use_cache: true

Show me the narrative, insights, and recommendations!
```

---

## Testing Guides

### For Quick Testing
👉 **[PHASE5_QUICK_TEST.md](./PHASE5_QUICK_TEST.md)**
- Copy/paste prompts
- 7 test suites
- Takes 10-15 minutes
- Covers all major features

### For Comprehensive Testing
👉 **[TESTING_PHASE5.md](./TESTING_PHASE5.md)**
- Detailed test scenarios
- Success criteria
- Performance benchmarks
- Debugging tips
- Takes 30-60 minutes

---

## Available Tools

### Updated Tools (Phase 5)

**`run_orchestrated_analysis`** - Enhanced with LLM support
```json
{
  "query": "natural language query",
  "use_llm": false,  // true = LLM mode, false = rule-based (default)
  "use_cache": true  // Enable query caching
}
```

Returns:
- Standard fields: lenses_executed, insights, recommendations
- **New in Phase 5**: narrative (if use_llm=true), cache_hit, cache_stats

### New Tools (Phase 5)

**`run_conversational_analysis`** - Multi-turn conversations
```json
{
  "query": "follow-up question",
  "use_llm": true,
  "conversation_history": [...]  // From previous turns
}
```

Returns:
- Same as orchestrated_analysis
- **Plus**: conversation_history, conversation_turn, token_usage

---

## Feature Comparison

| Feature | Rule-Based (use_llm=false) | LLM-Powered (use_llm=true) |
|---------|---------------------------|---------------------------|
| Query parsing | Keyword matching | Natural language understanding |
| Result synthesis | Simple aggregation | Coherent narratives |
| Conversation support | No | Yes (with history) |
| Cost per query | $0.00 | ~$0.05-0.10 (cold), $0.00 (cached) |
| Latency | <500ms | 2-5s (cold), <500ms (cached) |
| Requires API key | No ✅ | Yes |

---

## Cost Analysis

### Expected Costs (Claude 3.5 Sonnet)

**Per Query**:
- Query parsing: ~150 input + ~80 output tokens = ~$0.01-0.02
- Result synthesis: ~800 input + ~200 output tokens = ~$0.03-0.05
- **Total per cold query**: ~$0.05-0.10
- **Cached query**: $0.00

**With Caching** (typical usage):
- First query: $0.10
- Repeat queries: $0.00
- **Average with 50% hit rate**: $0.05/query

**Daily Usage Example**:
- 100 queries/day
- 50% cache hit rate
- Daily cost: ~$5.00
- Monthly cost: ~$150

**Cost Optimization Tips**:
1. Enable caching (use_cache=true)
2. Use rule-based mode for simple queries
3. Monitor token_usage field
4. Adjust cache TTL for your usage patterns

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         Four Lenses Analytics MCP Server            │
│                   Version 2.0.0                     │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
   ┌────▼────┐                    ┌────▼────┐
   │ Phase 3 │                    │ Phase 5 │
   │ Rule-   │                    │  LLM    │
   │ Based   │                    │ Powered │
   └────┬────┘                    └────┬────┘
        │                               │
        │  use_llm=false                │  use_llm=true
        │                               │
   ┌────▼────────────────┐    ┌────────▼─────────────┐
   │ Keyword Matching    │    │ QueryInterpreter     │
   │ Simple Aggregation  │    │ ResultSynthesizer    │
   │                     │    │ QueryCache           │
   └─────────────────────┘    └──────────────────────┘
```

**Hybrid Design Benefits**:
- Works immediately (no API key needed)
- Upgrade path (add API key when ready)
- Cost control (use LLM selectively)
- Fallback safety (LLM → rules on error)

---

## Files Added in Phase 5

**Core Components**:
- `orchestration/query_interpreter.py` - Claude-powered intent parsing
- `orchestration/result_synthesizer.py` - Narrative generation
- `orchestration/query_cache.py` - LRU cache with TTL
- `tools/conversational_analysis.py` - Multi-turn conversation tool

**Tests**:
- `tests/services/mcp_server/test_phase5_natural_language.py` - 21 tests

**Documentation**:
- `TESTING_PHASE5.md` - Comprehensive test guide
- `PHASE5_QUICK_TEST.md` - Quick-start prompts
- `README_PHASE5.md` - This file

---

## Troubleshooting

### "No module named 'anthropic'"
```bash
cd /path/to/AutoCLV
uv sync
```

### "Anthropic API key required"
1. Get key: https://console.anthropic.com/
2. Add to Claude Desktop config
3. Restart Claude Desktop

### Queries are slow
- First LLM query: 2-5s (normal)
- Cached queries: <500ms (instant)
- Use rule-based mode for speed: `use_llm: false`

### Cache not working
- Queries must be **identical** (case-sensitive)
- Ensure `use_cache: true`
- Check `cache_stats.hit_rate` in response

### Narrative is missing
- Only present when `use_llm: true`
- Check API key is valid
- Look for errors in response

### Still have issues?
1. Check MCP server logs: `~/Library/Logs/Claude/mcp*.log`
2. Run tests: `python -m pytest tests/services/mcp_server/test_phase5_natural_language.py -v`
3. Report issues: GitHub Issue #98

---

## Performance Benchmarks

Run the test suite and record your results:

| Test | Target | Your Result |
|------|--------|-------------|
| Rule-based query | <500ms | ___ms |
| LLM cold query | <5s | ___ms |
| LLM cached query | <500ms | ___ms |
| Cost per cold query | <$0.10 | $____ |
| Cache hit rate (10 queries) | >30% | ___% |

---

## Next Steps

1. ✅ Complete setup (API key, dependencies)
2. ✅ Run [Quick Test](./PHASE5_QUICK_TEST.md)
3. ⏸️ Run [Comprehensive Tests](./TESTING_PHASE5.md)
4. ⏸️ Document results in Issue #98
5. ⏸️ Deploy to production (optional)
6. ⏸️ Monitor costs and optimize caching

---

## Related Documentation

- **Issue #98**: [Phase 5: Natural Language Interface](https://github.com/datablogin/AutoCLV/issues/98)
- **Plan**: `thoughts/shared/plans/2025-10-14-agentic-five-lenses-implementation.md`
- **Phase 3**: Rule-based orchestration (baseline)
- **Phase 4A**: Observability & resilience
- **Anthropic Docs**: https://docs.anthropic.com/

---

## Support

Questions? Issues? Feedback?

- 📝 Open issue: https://github.com/datablogin/AutoCLV/issues
- 📖 Read tests: `tests/services/mcp_server/test_phase5_natural_language.py`
- 🔍 Check logs: `~/Library/Logs/Claude/mcp*.log`

---

**Version**: 2.0.0
**Status**: ✅ Complete
**Last Updated**: October 17, 2025
**Test Coverage**: 21 tests (all passing)
