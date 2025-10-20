"""Result Synthesizer - Phase 5

This module provides LLM-powered result synthesis using Claude.
It aggregates multi-lens results into coherent narratives with executive summaries,
key insights, and actionable recommendations.

Design:
- Uses Claude API (Anthropic SDK) for result synthesis
- Generates executive summary, insights, recommendations, and detailed narrative
- Handles partial results (when some lenses fail)
- Provides cost tracking via token usage monitoring
"""

import asyncio
import json
from typing import Any

import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class SynthesizedResults(BaseModel):
    """Synthesized analysis results from multiple lenses."""

    summary: str = Field(description="Executive summary (2-3 sentences)")
    insights: list[str] = Field(description="Key insights (3-5 bullet points)")
    recommendations: list[str] = Field(
        description="Actionable recommendations (3-5 bullet points)"
    )
    narrative: str = Field(description="Detailed narrative explanation")


class ResultSynthesizer:
    """Synthesizes multi-lens results using Claude.

    This class uses Claude to aggregate results from multiple lenses and
    generate coherent narratives that help users understand the overall
    state of their customer base.

    Cost optimization:
    - Uses efficient prompts to minimize token usage
    - Target: <1500 input tokens, <800 output tokens per synthesis
    - Estimated cost: ~$0.03-0.05 per synthesis (Claude 3.5 Sonnet)
    """

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize result synthesizer.

        Args:
            api_key: Anthropic API key
            model: Claude model to use (default: claude-3-5-sonnet-20241022)
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        logger.info("result_synthesizer_initialized", model=model)

    async def synthesize(
        self, query: str, lens_results: dict[str, dict[str, Any] | None]
    ) -> SynthesizedResults:
        """Synthesize results from multiple lenses into coherent narrative.

        This method uses Claude to:
        1. Analyze results from all executed lenses
        2. Generate an executive summary
        3. Extract key insights across lenses
        4. Provide actionable recommendations
        5. Create a detailed narrative explanation

        Args:
            query: Original user query
            lens_results: Dict mapping lens names (lens1, lens2, etc.) to their results.
                         None values indicate failed or skipped lenses.

        Returns:
            SynthesizedResults with summary, insights, recommendations, and narrative

        Raises:
            ValueError: If synthesis fails or Claude returns invalid JSON
        """
        logger.info(
            "synthesizing_results_with_claude",
            query=query,
            lens_count=sum(1 for v in lens_results.values() if v is not None),
            model=self.model,
        )

        # Filter out None results (failed/skipped lenses)
        valid_results = {k: v for k, v in lens_results.items() if v is not None}

        if not valid_results:
            logger.warning("no_valid_lens_results_for_synthesis")
            # Return minimal synthesis when no results available
            return SynthesizedResults(
                summary="No lens results available for synthesis.",
                insights=["Analysis could not be completed due to missing data."],
                recommendations=[
                    "Ensure foundation data is loaded (transactions, data mart, RFM, cohorts)."
                ],
                narrative="Unable to generate detailed analysis without lens results. "
                "Please verify that required data is loaded and try again.",
            )

        prompt = self._build_prompt(query, valid_results)

        try:
            # Add 30-second timeout to prevent indefinite hangs
            response = await asyncio.wait_for(
                self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
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
            synthesis_dict = self._extract_json(content)

            # Validate and construct SynthesizedResults
            synthesized = SynthesizedResults(**synthesis_dict)

            logger.info(
                "synthesis_complete",
                insight_count=len(synthesized.insights),
                recommendation_count=len(synthesized.recommendations),
                summary_length=len(synthesized.summary),
                narrative_length=len(synthesized.narrative),
            )

            return synthesized

        except Exception as e:
            logger.error(
                "synthesis_failed",
                error=str(e),
                error_type=type(e).__name__,
                query=query,
            )
            raise ValueError(f"Failed to synthesize results: {str(e)}") from e

    def _build_prompt(self, query: str, lens_results: dict[str, dict[str, Any]]) -> str:
        """Build prompt for Claude API.

        Constructs a comprehensive prompt that includes the original query
        and all lens results formatted for Claude to analyze.

        Args:
            query: Original user query
            lens_results: Valid lens results (no None values)

        Returns:
            Formatted prompt string
        """
        # Format lens results as readable text
        results_text = self._format_results(lens_results)

        return f"""You are an expert customer analytics consultant synthesizing analysis results.

Original query: "{query}"

Analysis results:
{results_text}

Provide a comprehensive synthesis with:
1. Executive summary (2-3 sentences capturing the most important findings)
2. Key insights (3-5 bullet points highlighting critical patterns and trends)
3. Actionable recommendations (3-5 bullet points with specific next steps)
4. Detailed narrative explanation (comprehensive analysis connecting all lenses)

Return JSON with this structure:
{{
  "summary": "Executive summary...",
  "insights": ["Insight 1...", "Insight 2...", "Insight 3..."],
  "recommendations": ["Recommendation 1...", "Recommendation 2...", "Recommendation 3..."],
  "narrative": "Detailed narrative explanation..."
}}

Guidelines:
- Focus on business impact, not just metrics
- Prioritize insights by importance (most critical first)
- Make recommendations specific and actionable
- Connect insights across lenses to show the full picture
- Use clear, non-technical language where possible
- Highlight both strengths and areas for improvement"""

    def _format_results(self, lens_results: dict[str, dict[str, Any]]) -> str:
        """Format lens results as readable text for Claude.

        Converts lens result dictionaries into human-readable format that
        Claude can easily analyze.

        Args:
            lens_results: Valid lens results (no None values)

        Returns:
            Formatted results string
        """
        formatted_lines = []

        for lens_name, result in sorted(lens_results.items()):
            formatted_lines.append(f"\n{lens_name.upper()} RESULTS:")

            # Format key metrics based on lens type
            if lens_name == "lens1":
                formatted_lines.append(
                    f"  - Total Customers: {result.get('total_customers', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - One-Time Buyer %: {result.get('one_time_buyer_pct', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Health Score: {result.get('customer_health_score', 'N/A')}/100"
                )
                formatted_lines.append(
                    f"  - Concentration Risk: {result.get('concentration_risk', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Top 10% Revenue Contribution: {result.get('top_10pct_revenue_contribution', 'N/A')}%"
                )

            elif lens_name == "lens2":
                formatted_lines.append(
                    f"  - Retention Rate: {result.get('retention_rate', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Churn Rate: {result.get('churn_rate', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Growth Momentum: {result.get('growth_momentum', 'N/A')}"
                )

            elif lens_name == "lens3":
                formatted_lines.append(
                    f"  - Cohort ID: {result.get('cohort_id', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Cohort Size: {result.get('cohort_size', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Maturity: {result.get('cohort_maturity', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - LTV Trajectory: {result.get('ltv_trajectory', 'N/A')}"
                )

            elif lens_name == "lens4":
                formatted_lines.append(
                    f"  - Cohort Count: {result.get('cohort_count', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Best Cohort: {result.get('best_performing_cohort', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Worst Cohort: {result.get('worst_performing_cohort', 'N/A')}"
                )

            elif lens_name == "lens5":
                formatted_lines.append(
                    f"  - Health Score: {result.get('health_score', 'N/A')}/100"
                )
                formatted_lines.append(
                    f"  - Health Grade: {result.get('health_grade', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Total Customers: {result.get('total_customers', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Overall Retention Rate: {result.get('overall_retention_rate', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Cohort Quality Trend: {result.get('cohort_quality_trend', 'N/A')}"
                )
                formatted_lines.append(
                    f"  - Revenue Predictability: {result.get('revenue_predictability_pct', 'N/A')}%"
                )

            # Include insights and recommendations if available
            if "insights" in result and result["insights"]:
                formatted_lines.append("  Insights:")
                for insight in result["insights"][:3]:  # Limit to top 3
                    formatted_lines.append(f"    - {insight}")

            if "recommendations" in result and result["recommendations"]:
                formatted_lines.append("  Recommendations:")
                for rec in result["recommendations"][:3]:  # Limit to top 3
                    formatted_lines.append(f"    - {rec}")

        return "\n".join(formatted_lines)

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

        # Parse JSON (strict=False allows control characters like newlines in strings)
        try:
            return json.loads(content.strip(), strict=False)
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
