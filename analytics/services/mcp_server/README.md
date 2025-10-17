# Four Lenses Analytics MCP Server

Phase 3 implementation of the LangGraph-based orchestration layer for AutoCLV Four Lenses customer analytics.

## Version

**Current Version**: 0.2.0
**Phase**: Phase 3 - LangGraph Orchestration
**Status**: Production-ready for testing

## Features

### Foundation Services (Phase 1)
- `build_customer_data_mart`: Build customer data mart from transaction data
- `calculate_rfm_metrics`: Calculate RFM (Recency, Frequency, Monetary) metrics
- `create_customer_cohorts`: Create and assign customer cohorts

### Lens Services (Phase 2)
- `analyze_single_period_snapshot`: Lens 1 - Single period health analysis
- `analyze_period_to_period_comparison`: Lens 2 - Period comparison (placeholder)
- `analyze_cohort_lifecycle`: Lens 3 - Single cohort evolution (placeholder)
- `compare_multiple_cohorts`: Lens 4 - Multi-cohort comparison (placeholder)
- `analyze_overall_customer_base_health`: Lens 5 - Overall base health

### Orchestration (Phase 3) ✨ NEW
- `run_orchestrated_analysis`: Orchestrated multi-lens analysis with:
  - Rule-based intent parsing
  - Parallel lens execution (Lens 1, 3, 4, 5)
  - Dependency-aware sequencing (Lens 2 after Lens 1)
  - Result aggregation and synthesis
  - Execution time tracking

### Health & Monitoring
- `health_check`: Server health status

## Setup with Claude Desktop

### 1. Install Claude Desktop
Download from: https://claude.ai/download

### 2. Configure MCP Server

#### macOS
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
        "PYTHONPATH": "/Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a"
      }
    }
  }
}
```

#### Linux
Edit `~/.config/claude/claude_desktop_config.json` with the same content (adjust path as needed).

### 3. Restart Claude Desktop
Close and reopen Claude Desktop to load the MCP server.

### 4. Verify Connection
In Claude Desktop, type:
```
What tools are available?
```

You should see all 9 Four Lenses Analytics tools listed.

## Usage Examples

### Basic Health Check
```
Run health_check
```

### Foundation Workflow
```
1. build_customer_data_mart with transaction data
2. calculate_rfm_metrics
3. create_customer_cohorts
```

### Individual Lens Analysis
```
Run analyze_single_period_snapshot for customer health
```

### Orchestrated Analysis (Phase 3) 🎯

**Simple queries:**
```
customer health snapshot
overall customer base health
show me lens1
```

**Multi-lens queries:**
```
customer health and cohort analysis
lens1 and lens5
overall health
```

**Expected output:**
- List of executed lenses
- Aggregated insights
- Actionable recommendations
- Execution time
- Individual lens results
- Foundation data status

## Architecture

### State Management
The server uses `SharedState` for cross-tool data persistence:
- Data mart
- RFM metrics and scores
- Cohort definitions and assignments

### Orchestration Flow
```
Query → Intent Parsing → Foundation Check → Parallel Lens Execution → Synthesis
```

**Parallel Groups:**
- Group 1: Lens 1, 3, 4, 5 (independent, run concurrently)
- Group 2: Lens 2 (depends on Lens 1, sequential)

### Error Handling
- Graceful degradation: Partial results returned if some lenses fail
- Missing data warnings: Foundation readiness checks
- Detailed error logging: Structured logging with execution context

## Development

### Run Server Directly
```bash
cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
uv run fastmcp dev analytics/services/mcp_server/main.py
```

### Run Tests
```bash
pytest tests/services/mcp_server/test_orchestration.py -v
```

### Lint and Format
```bash
ruff check analytics/services/mcp_server/
ruff format analytics/services/mcp_server/
```

## Troubleshooting

### Server Not Connecting
1. Check Claude Desktop logs (Help → View Logs)
2. Verify `uv` is installed: `which uv`
3. Verify Python environment: `uv run python --version`
4. Test server manually: `uv run fastmcp dev analytics/services/mcp_server/main.py`

### Tools Not Appearing
1. Verify config file syntax (valid JSON)
2. Check file paths are absolute
3. Restart Claude Desktop completely
4. Check stderr output for errors

### Orchestrated Analysis Fails
1. Ensure foundation data is prepared:
   - `build_customer_data_mart` first
   - `calculate_rfm_metrics` for Lens 1/2
   - `create_customer_cohorts` for Lens 3/4/5
2. Check `lenses_failed` in response for specific errors
3. Review `SharedState` for missing data

### Lens Execution Errors
Check the logs in stderr for detailed error messages. Common issues:
- Missing RFM data: Run `calculate_rfm_metrics`
- Missing cohort data: Run `create_customer_cohorts`
- Invalid data formats: Check input data structure

## Next Steps (Phase 4 & 5)

### Phase 4: Enhanced Observability (Optional)
- [ ] Retry logic with exponential backoff
- [ ] Health check MCP tool enhancements
- [ ] Execution metrics tracking
- [ ] OpenTelemetry OTLP export (production)

### Phase 5: Natural Language Interface (Optional)
- [ ] LLM-based intent parsing with Claude
- [ ] LLM-based result synthesis
- [ ] Conversational context maintenance
- [ ] Cost and latency optimization

## License

Internal project - AutoCLV Analytics Platform

## Contact

For issues or questions, contact the AutoCLV development team.

---

**Phase 3 Implementation Date**: 2025-10-16
**Contributors**: Claude Code + Human Review
**Test Coverage**: 8 orchestration tests (100% passing)
