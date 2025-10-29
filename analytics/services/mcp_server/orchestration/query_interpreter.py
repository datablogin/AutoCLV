"""Query Interpreter - Phase 5

This module provides LLM-powered natural language query interpretation using Claude.
It translates user queries into structured intents that specify which lenses to run
and with what parameters.

Design:
- Uses Claude API (Anthropic SDK) for query parsing
- Returns structured JSON with lenses, date ranges, filters, and reasoning
- Handles ambiguous queries with intelligent defaults
- Provides cost tracking via token usage monitoring
"""

import asyncio
import json
from typing import Any

import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class ParsedIntent(BaseModel):
    """Structured intent parsed from natural language query."""

    lenses: list[str] = Field(
        description="List of lenses to execute (lens1, lens2, lens3, lens4, lens5)"
    )
    date_range: dict[str, str] | None = Field(
        default=None,
        description="Date range with 'start' and 'end' keys (ISO format)",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Additional filters (cohort_id, etc.)"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Additional analysis parameters"
    )
    reasoning: str = Field(description="Explanation of why these lenses were selected")


class QueryInterpreter:
    """Interprets natural language queries using Claude.

    This class uses Claude to parse user queries and extract structured intents
    that specify which lenses to run, what date ranges to analyze, and any
    additional filters or parameters.

    Cost optimization:
    - Uses efficient prompts to minimize token usage
    - Target: <500 input tokens, <200 output tokens per query
    - Estimated cost: ~$0.01-0.02 per query (Claude 3.5 Sonnet)
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize query interpreter.

        Args:
            api_key: Anthropic API key
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        logger.info("query_interpreter_initialized", model=model)

    async def parse_query(self, query: str) -> ParsedIntent:
        """Parse natural language query into structured intent.

        This method uses Claude to analyze the user's query and determine:
        1. Which lenses should be executed
        2. What date ranges to analyze (if specified)
        3. Any cohort or segment filters
        4. Additional analysis parameters

        Args:
            query: Natural language query from user

        Returns:
            ParsedIntent with structured analysis plan

        Raises:
            ValueError: If Claude returns invalid JSON or parsing fails
        """
        # Query is already sanitized by coordinator.analyze() or tool level
        # No need to sanitize again here to avoid double sanitization
        logger.info("parsing_query_with_claude", query=query, model=self.model)

        prompt = self._build_prompt(query)

        try:
            # Add 30-second timeout to prevent indefinite hangs
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=30.0,
            )

            # Track token usage for cost monitoring
            usage = response.usage
            self._total_input_tokens += usage.input_tokens
            self._total_output_tokens += usage.output_tokens

            logger.info(
                "claude_api_response_received",
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_input_tokens=self._total_input_tokens,
                total_output_tokens=self._total_output_tokens,
            )

            # Extract JSON from response
            content = response.content[0].text
            intent_dict = self._extract_json(content)

            # Validate and construct ParsedIntent
            parsed_intent = ParsedIntent(**intent_dict)

            logger.info(
                "intent_parsed_successfully",
                lenses=parsed_intent.lenses,
                lens_count=len(parsed_intent.lenses),
                has_date_range=parsed_intent.date_range is not None,
                has_filters=bool(parsed_intent.filters),
                reasoning=parsed_intent.reasoning[:100],  # Truncate for logging
            )

            return parsed_intent

        except Exception as e:
            logger.error(
                "query_parsing_failed",
                error=str(e),
                error_type=type(e).__name__,
                query=query,
            )
            raise ValueError(f"Failed to parse query: {str(e)}") from e

    def _build_prompt(self, query: str) -> str:
        """Build prompt for Claude API.

        Constructs a concise but comprehensive prompt that guides Claude to
        extract the correct intent from the user's query.

        Args:
            query: User's natural language query

        Returns:
            Formatted prompt string
        """
        return f"""You are an expert in customer analytics using the Five Lenses methodology.

Parse this user query into a structured analysis plan:

Query: "{query}"

Available lenses:
- Lens 1: Single-period snapshot (current customer base health, RFM distribution, revenue concentration)
- Lens 2: Period-to-period comparison (retention, churn, growth momentum, customer migration)
- Lens 3: Single cohort evolution (lifecycle of one acquisition cohort, LTV trajectory)
- Lens 4: Multi-cohort comparison (which acquisition cohorts perform best, cohort quality)
- Lens 5: Overall customer base health (comprehensive health scoring, cohort trends, predictability)

Return JSON with this structure:
{{
  "lenses": ["lens1"],  // Which lenses to run (lens1, lens2, lens3, lens4, lens5)
  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},  // null if not specified
  "filters": {{"cohort_id": "2024-Q1"}},  // Any cohort/segment filters (empty dict if none)
  "parameters": {{}},  // Additional parameters (empty dict if none)
  "reasoning": "Why these lenses were selected and how they address the query"
}}

Guidelines:
- "health" or "snapshot" → lens1
- "overall health" or "base health" → lens5
- "compare" or "trend" → lens2
- "cohort evolution" or "lifecycle" → lens3
- "cohort comparison" or "which cohort" → lens4
- If query is ambiguous, default to lens1
- If multiple aspects requested, include multiple lenses
- Be precise with lens selection based on query intent"""

    def _extract_json(self, content: str) -> dict[str, Any]:
        """Extract JSON from Claude response.

        Claude may return JSON wrapped in markdown code blocks or with
        additional text. This method extracts the raw JSON.

        Args:
            content: Raw response text from Claude

        Returns:
            Parsed JSON as dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        # Remove markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        # Parse JSON
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(
                "json_parsing_failed",
                content=content[:200],  # Truncate for logging
                error=str(e),
            )
            raise ValueError(f"Invalid JSON in Claude response: {str(e)}") from e

    def get_token_usage(self) -> dict[str, int]:
        """Get cumulative token usage for cost tracking.

        Returns:
            Dictionary with input_tokens, output_tokens, and total_tokens
        """
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
        }

    def reset_token_usage(self) -> None:
        """Reset token usage counters."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        logger.info("token_usage_reset")
