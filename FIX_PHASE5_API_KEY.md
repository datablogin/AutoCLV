# Fix Phase 5 API Key Issue

## Problem
The API key is in `.env` but Claude Desktop can't read it. The MCP server needs the key in Claude Desktop's config.

## Solution

### Step 1: Open Claude Desktop Config

```bash
# Method 1: Open in TextEdit
open -e ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Method 2: Open in VS Code (if installed)
code ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Method 3: Manual path
# Navigate to: ~/Library/Application Support/Claude/claude_desktop_config.json
```

### Step 2: Add API Key to Config

Find the section for `"four-lenses-analytics"` and update the `"env"` section:

**Before:**
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
        "PYTHONPATH": "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a"
      }
    }
  }
}
```

**After:**
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
        "ANTHROPIC_API_KEY": "sk-ant-api03-YOUR_API_KEY_HERE"
      }
    }
  }
}
```

**Important:** Make sure there's a **comma** after the PYTHONPATH line!

### Step 3: Restart Claude Desktop

**Critical:** You MUST restart Claude Desktop for the change to take effect.

1. Quit Claude Desktop completely (⌘Q or Cmd+Q)
2. Wait 2-3 seconds
3. Reopen Claude Desktop
4. Wait for it to fully load (you'll see the MCP server connect)

### Step 4: Verify API Key is Working

Run this quick test in Claude Desktop:

```
Run run_orchestrated_analysis with:
- query: "test api key"
- use_llm: true
- use_cache: false
```

**Expected result:**
- ✅ Should work (no API key error)
- ✅ Should return a `narrative` field
- ✅ Should show `token_usage`

**If it still fails:**
- Check JSON syntax is valid (use jsonlint.com)
- Ensure comma after PYTHONPATH line
- Verify you restarted Claude Desktop
- Check API key copied exactly (no spaces)

---

## Once Working: Complete Phase 5 Testing

Run only the Phase 5 steps (Steps 9-13):

```
PHASE 5 COMPLETION TEST
========================

Step 9: First LLM query (cold - cache miss)
Run run_orchestrated_analysis with query "Tell me about customer health and which cohorts are performing best", use_llm true, use_cache true

Step 10: Cached LLM query (warm - cache hit)
Run run_orchestrated_analysis with query "Tell me about customer health and which cohorts are performing best", use_llm true, use_cache true

Step 11: Different LLM query
Run run_orchestrated_analysis with query "What's the retention trend and overall customer base health?", use_llm true, use_cache true

Step 12: Conversational analysis - Turn 1
Run run_conversational_analysis with query "Give me a comprehensive customer base analysis", use_llm true, conversation_history null

Step 13: Conversational analysis - Turn 2
Run run_conversational_analysis with query "Now focus on which cohorts need attention", use_llm true, conversation_history [paste result from Step 12]

Step 14: Final metrics
Run get_execution_metrics
```

---

## Expected Results

### Step 9 (First LLM Query)
- ⏱️ Duration: 2-5 seconds
- 💰 Cost: ~$0.05-0.10
- ✅ cache_hit: false
- ✅ narrative field present
- ✅ token_usage: ~700-1400 tokens

### Step 10 (Cached Query)
- ⚡ Duration: <100ms (instant!)
- 💰 Cost: $0.00
- ✅ cache_hit: true
- ✅ cache_stats.hit_rate: 0.5 (50%)
- ✅ Identical results to Step 9

### Step 11 (Different Query)
- ⏱️ Duration: 2-5 seconds
- 💰 Cost: ~$0.05-0.10
- ✅ cache_hit: false (different query)
- ✅ New narrative

### Steps 12-13 (Conversation)
- ⏱️ Each: 2-5 seconds
- 💰 Each: ~$0.05-0.10
- ✅ conversation_turn: 1, then 2
- ✅ Context maintained
- ✅ Cumulative token_usage

### Total Expected Cost
**$0.25-0.50** for all 5 LLM queries

---

## Success Criteria for Complete Phase 5

- [ ] All steps 9-13 complete without errors
- [ ] Step 9 generates narrative (not just insights)
- [ ] Step 10 shows cache_hit: true and is instant
- [ ] Token usage is reasonable (<1500 tokens/query)
- [ ] Total cost is under $0.50
- [ ] Conversational context maintained across turns
- [ ] Narrative quality exceeds rule-based insights

Once these pass, Phase 5 is **FULLY VALIDATED**! 🎉
