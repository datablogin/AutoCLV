"""Test formatters for Five Lenses analysis results."""

from datetime import datetime, timezone
from decimal import Decimal

from customer_base_audit.analyses.lens1 import Lens1Metrics
from customer_base_audit.analyses.lens2 import Lens2Metrics, CustomerMigration
from customer_base_audit.analyses.lens3 import Lens3Metrics, CohortPeriodMetrics
from customer_base_audit.analyses.lens4 import (
    Lens4Metrics,
    CohortDecomposition,
    TimeToSecondPurchase,
)
from customer_base_audit.analyses.lens5 import (
    Lens5Metrics,
    CustomerBaseHealthScore,
)
from customer_base_audit.mcp.formatters.markdown_tables import (
    format_lens1_table,
    format_lens2_table,
    format_lens4_decomposition_table,
    format_lens5_health_summary_table,
)
from customer_base_audit.mcp.formatters.plotly_charts import (
    create_retention_trend_chart,
    create_revenue_concentration_pie,
    create_health_score_gauge,
    create_executive_dashboard,
    create_cohort_heatmap,
    create_sankey_diagram,
)
from customer_base_audit.mcp.formatters.executive_summaries import (
    generate_health_summary,
    generate_retention_insights,
    generate_cohort_comparison,
)


class TestMarkdownTables:
    """Test markdown table formatters."""

    def test_format_lens1_table_basic(self):
        """Lens1 table should include key metrics."""
        metrics = Lens1Metrics(
            total_customers=100,
            one_time_buyers=30,
            one_time_buyer_pct=Decimal("30.00"),
            total_revenue=Decimal("10000.00"),
            top_10pct_revenue_contribution=Decimal("45.00"),
            top_20pct_revenue_contribution=Decimal("62.00"),
            avg_orders_per_customer=Decimal("2.50"),
            median_customer_value=Decimal("100.00"),
            rfm_distribution={},
        )

        table = format_lens1_table(metrics)

        assert "## Lens 1" in table
        assert "100" in table  # total customers
        assert "30.00%" in table  # one-time buyer pct
        assert "$10,000.00" in table  # total revenue
        assert "45.00%" in table  # top 10%

    def test_format_lens1_table_with_rfm(self):
        """Lens1 table should include RFM distribution if available."""
        metrics = Lens1Metrics(
            total_customers=100,
            one_time_buyers=30,
            one_time_buyer_pct=Decimal("30.00"),
            total_revenue=Decimal("10000.00"),
            top_10pct_revenue_contribution=Decimal("45.00"),
            top_20pct_revenue_contribution=Decimal("62.00"),
            avg_orders_per_customer=Decimal("2.50"),
            median_customer_value=Decimal("100.00"),
            rfm_distribution={"555": 10, "111": 5},
        )

        table = format_lens1_table(metrics)

        assert "RFM" in table
        assert "555" in table
        assert "111" in table

    def test_format_lens2_table(self):
        """Lens2 table should show migration and changes."""
        p1 = Lens1Metrics(
            100,
            40,
            Decimal("40.00"),
            Decimal("10000.00"),
            Decimal("45.00"),
            Decimal("62.00"),
            Decimal("2.50"),
            Decimal("100.00"),
            {},
        )
        p2 = Lens1Metrics(
            120,
            50,
            Decimal("41.67"),
            Decimal("12000.00"),
            Decimal("47.00"),
            Decimal("64.00"),
            Decimal("2.80"),
            Decimal("100.00"),
            {},
        )
        migration = CustomerMigration(
            retained=frozenset(["C1", "C2"]),
            churned=frozenset(["C3"]),
            new=frozenset(["C4", "C5", "C6"]),
            reactivated=frozenset(["C5"]),
        )
        metrics = Lens2Metrics(
            period1_metrics=p1,
            period2_metrics=p2,
            migration=migration,
            retention_rate=Decimal("66.67"),
            churn_rate=Decimal("33.33"),
            reactivation_rate=Decimal("33.33"),
            customer_count_change=20,
            revenue_change_pct=Decimal("20.00"),
            avg_order_value_change_pct=Decimal("5.00"),
        )

        table = format_lens2_table(metrics)

        assert "## Lens 2" in table
        assert "66.67%" in table  # retention rate
        assert "33.33%" in table  # churn rate
        assert "2" in table  # retained count
        assert "3" in table  # new count

    def test_format_lens4_decomposition_table(self):
        """Lens4 table should show cohort comparisons."""
        decomp = CohortDecomposition(
            cohort_id="2023-Q1",
            period_number=0,
            cohort_size=100,
            active_customers=100,
            pct_active=Decimal("100.00"),
            total_orders=150,
            aof=Decimal("1.50"),
            total_revenue=Decimal("15000.00"),
            aov=Decimal("100.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("15000.00"),
        )
        ttsp = TimeToSecondPurchase(
            cohort_id="2023-Q1",
            customers_with_repeat=60,
            repeat_rate=Decimal("60.00"),
            median_days=Decimal("30.00"),
            mean_days=Decimal("35.00"),
            cumulative_repeat_by_period={},
        )
        metrics = Lens4Metrics(
            cohort_decompositions=[decomp],
            time_to_second_purchase=[ttsp],
            cohort_comparisons=[],
            alignment_type="left-aligned",
        )

        table = format_lens4_decomposition_table(metrics)

        assert "## Lens 4" in table
        assert "2023-Q1" in table
        assert "60.00%" in table  # repeat rate
        assert "Left-Aligned" in table

    def test_format_lens5_health_summary_table(self):
        """Lens5 table should show health scorecard."""
        health = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=800,
            overall_retention_rate=Decimal("80.00"),
            cohort_quality_trend="improving",
            revenue_predictability_pct=Decimal("85.00"),
            acquisition_dependence_pct=Decimal("15.00"),
            health_score=Decimal("82.50"),
            health_grade="B",
        )
        metrics = Lens5Metrics(
            cohort_revenue_contributions=[],
            cohort_repeat_behavior=[],
            health_score=health,
            analysis_start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            analysis_end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
        )

        table = format_lens5_health_summary_table(metrics)

        assert "## Lens 5" in table
        assert "Grade B" in table
        assert "82.50" in table
        assert "80.00%" in table  # retention
        assert "📈" in table  # improving trend emoji


class TestPlotlyCharts:
    """Test Plotly chart generators."""

    def test_create_retention_trend_chart(self):
        """Retention trend chart should have proper structure."""
        metrics = Lens3Metrics(
            cohort_name="2023-Q1",
            acquisition_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            cohort_size=100,
            periods=[
                CohortPeriodMetrics(0, 100, 1.0, 1.5, 50.0, 1.5, 50.0, 5000.0),
                CohortPeriodMetrics(1, 80, 0.9, 1.2, 40.0, 0.96, 32.0, 3200.0),
                CohortPeriodMetrics(2, 60, 0.85, 1.0, 35.0, 0.60, 21.0, 2100.0),
            ],
        )

        result = create_retention_trend_chart(metrics)

        # Verify dual format (PNG + JSON)
        assert "plotly_json" in result
        assert "image_base64" in result
        assert result["format"] == "png"

        chart = result["plotly_json"]
        assert "data" in chart
        assert "layout" in chart
        assert len(chart["data"]) == 2  # retention line + active bars
        assert chart["data"][0]["type"] == "scatter"
        assert chart["data"][1]["type"] == "bar"
        assert "2023-Q1" in chart["layout"]["title"]["text"]

    def test_create_revenue_concentration_pie(self):
        """Revenue concentration pie should show Pareto distribution."""
        metrics = Lens1Metrics(
            total_customers=100,
            one_time_buyers=30,
            one_time_buyer_pct=Decimal("30.00"),
            total_revenue=Decimal("10000.00"),
            top_10pct_revenue_contribution=Decimal("45.00"),
            top_20pct_revenue_contribution=Decimal("62.00"),
            avg_orders_per_customer=Decimal("2.50"),
            median_customer_value=Decimal("100.00"),
            rfm_distribution={},
        )

        result = create_revenue_concentration_pie(metrics)

        # Verify dual format (PNG + JSON)
        assert "plotly_json" in result
        assert "image_base64" in result
        assert result["format"] == "png"

        chart = result["plotly_json"]
        assert "data" in chart
        assert "layout" in chart
        assert chart["data"][0]["type"] == "pie"
        assert len(chart["data"][0]["labels"]) == 3  # top 10%, next 10%, remaining 80%
        assert chart["data"][0]["values"][0] == 45.0  # top 10%

    def test_create_health_score_gauge(self):
        """Health score gauge should be an indicator."""
        health = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=800,
            overall_retention_rate=Decimal("80.00"),
            cohort_quality_trend="improving",
            revenue_predictability_pct=Decimal("85.00"),
            acquisition_dependence_pct=Decimal("15.00"),
            health_score=Decimal("82.50"),
            health_grade="B",
        )
        metrics = Lens5Metrics(
            cohort_revenue_contributions=[],
            cohort_repeat_behavior=[],
            health_score=health,
            analysis_start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            analysis_end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
        )

        result = create_health_score_gauge(metrics)

        # Verify dual format (PNG + JSON)
        assert "plotly_json" in result
        assert "image_base64" in result
        assert result["format"] == "png"

        chart = result["plotly_json"]
        assert "data" in chart
        assert "layout" in chart
        assert chart["data"][0]["type"] == "indicator"
        assert chart["data"][0]["value"] == 82.5
        assert "Grade B" in chart["data"][0]["title"]["text"]

    def test_create_executive_dashboard(self):
        """Executive dashboard should combine multiple metrics."""
        lens1 = Lens1Metrics(
            total_customers=100,
            one_time_buyers=30,
            one_time_buyer_pct=Decimal("30.00"),
            total_revenue=Decimal("10000.00"),
            top_10pct_revenue_contribution=Decimal("45.00"),
            top_20pct_revenue_contribution=Decimal("62.00"),
            avg_orders_per_customer=Decimal("2.50"),
            median_customer_value=Decimal("100.00"),
            rfm_distribution={},
        )
        health = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=800,
            overall_retention_rate=Decimal("80.00"),
            cohort_quality_trend="improving",
            revenue_predictability_pct=Decimal("85.00"),
            acquisition_dependence_pct=Decimal("15.00"),
            health_score=Decimal("82.50"),
            health_grade="B",
        )
        lens5 = Lens5Metrics(
            cohort_revenue_contributions=[],
            cohort_repeat_behavior=[],
            health_score=health,
            analysis_start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            analysis_end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
        )

        result = create_executive_dashboard(lens1, lens5)

        # Verify dual format (PNG + JSON)
        assert "plotly_json" in result
        assert "image_base64" in result
        assert result["format"] == "png"

        dashboard = result["plotly_json"]
        assert "data" in dashboard
        assert "layout" in dashboard
        assert len(dashboard["data"]) == 6  # 4 KPIs + pie + bar
        assert "Dashboard" in dashboard["layout"]["title"]["text"]

    def test_create_cohort_heatmap(self):
        """Cohort heatmap should visualize retention across cohorts and periods."""
        decomp1 = CohortDecomposition(
            cohort_id="2023-Q1",
            period_number=0,
            cohort_size=100,
            active_customers=100,
            pct_active=Decimal("100.00"),
            total_orders=150,
            aof=Decimal("1.50"),
            total_revenue=Decimal("15000.00"),
            aov=Decimal("100.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("15000.00"),
        )
        decomp2 = CohortDecomposition(
            cohort_id="2023-Q1",
            period_number=1,
            cohort_size=100,
            active_customers=80,
            pct_active=Decimal("80.00"),
            total_orders=120,
            aof=Decimal("1.50"),
            total_revenue=Decimal("12000.00"),
            aov=Decimal("100.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("12000.00"),
        )
        decomp3 = CohortDecomposition(
            cohort_id="2023-Q2",
            period_number=0,
            cohort_size=120,
            active_customers=120,
            pct_active=Decimal("100.00"),
            total_orders=180,
            aof=Decimal("1.50"),
            total_revenue=Decimal("18000.00"),
            aov=Decimal("100.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("18000.00"),
        )
        metrics = Lens4Metrics(
            cohort_decompositions=[decomp1, decomp2, decomp3],
            time_to_second_purchase=[],
            cohort_comparisons=[],
            alignment_type="left-aligned",
        )

        result = create_cohort_heatmap(metrics)

        # Verify dual format (PNG + JSON)
        assert "plotly_json" in result
        assert "image_base64" in result
        assert result["format"] == "png"

        chart = result["plotly_json"]
        assert "data" in chart
        assert "layout" in chart
        assert chart["data"][0]["type"] == "heatmap"
        assert len(chart["data"][0]["y"]) == 2  # 2 cohorts
        assert "Left-Aligned" in chart["layout"]["title"]["text"]
        assert chart["data"][0]["zmin"] == 0
        assert chart["data"][0]["zmax"] == 100

    def test_create_sankey_diagram(self):
        """Sankey diagram should show customer migration flow."""
        p1 = Lens1Metrics(
            total_customers=100,
            one_time_buyers=40,
            one_time_buyer_pct=Decimal("40.00"),
            total_revenue=Decimal("10000.00"),
            top_10pct_revenue_contribution=Decimal("45.00"),
            top_20pct_revenue_contribution=Decimal("62.00"),
            avg_orders_per_customer=Decimal("2.50"),
            median_customer_value=Decimal("100.00"),
            rfm_distribution={},
        )
        p2 = Lens1Metrics(
            total_customers=120,
            one_time_buyers=50,
            one_time_buyer_pct=Decimal("41.67"),
            total_revenue=Decimal("12000.00"),
            top_10pct_revenue_contribution=Decimal("47.00"),
            top_20pct_revenue_contribution=Decimal("64.00"),
            avg_orders_per_customer=Decimal("2.80"),
            median_customer_value=Decimal("100.00"),
            rfm_distribution={},
        )
        migration = CustomerMigration(
            retained=frozenset(["C1", "C2", "C3"]),
            churned=frozenset(["C4"]),
            new=frozenset(["C5", "C6", "C7", "C8"]),
            reactivated=frozenset(["C8"]),  # Must be subset of new
        )
        metrics = Lens2Metrics(
            period1_metrics=p1,
            period2_metrics=p2,
            migration=migration,
            retention_rate=Decimal("75.00"),
            churn_rate=Decimal("25.00"),
            reactivation_rate=Decimal("25.00"),
            customer_count_change=20,
            revenue_change_pct=Decimal("20.00"),
            avg_order_value_change_pct=Decimal("5.00"),
        )

        result = create_sankey_diagram(metrics)

        # Verify dual format (PNG + JSON)
        assert "plotly_json" in result
        assert "image_base64" in result
        assert result["format"] == "png"

        chart = result["plotly_json"]
        assert "data" in chart
        assert "layout" in chart
        assert chart["data"][0]["type"] == "sankey"
        assert "node" in chart["data"][0]
        assert "link" in chart["data"][0]
        assert len(chart["data"][0]["node"]["label"]) == 6  # 6 nodes
        # Should have flows: retained (2 links), churned (1), new (1), reactivated (1) = 5 total
        assert len(chart["data"][0]["link"]["source"]) == 5
        assert "Migration Flow" in chart["layout"]["title"]["text"]


class TestExecutiveSummaries:
    """Test executive summary generators."""

    def test_generate_health_summary_grade_a(self):
        """Health summary for grade A should be positive."""
        health = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=900,
            overall_retention_rate=Decimal("90.00"),
            cohort_quality_trend="improving",
            revenue_predictability_pct=Decimal("90.00"),
            acquisition_dependence_pct=Decimal("10.00"),
            health_score=Decimal("92.00"),
            health_grade="A",
        )
        metrics = Lens5Metrics(
            cohort_revenue_contributions=[],
            cohort_repeat_behavior=[],
            health_score=health,
            analysis_start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            analysis_end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
        )

        summary = generate_health_summary(metrics)

        assert "Overall Grade: A" in summary
        assert "excellent" in summary
        assert "92.00" in summary
        assert "90.00%" in summary

    def test_generate_health_summary_with_warning(self):
        """Health summary should warn on high acquisition dependence."""
        health = CustomerBaseHealthScore(
            total_customers=1000,
            total_active_customers=800,
            overall_retention_rate=Decimal("80.00"),
            cohort_quality_trend="improving",
            revenue_predictability_pct=Decimal("65.00"),
            acquisition_dependence_pct=Decimal("35.00"),
            health_score=Decimal("75.00"),
            health_grade="C",
        )
        metrics = Lens5Metrics(
            cohort_revenue_contributions=[],
            cohort_repeat_behavior=[],
            health_score=health,
            analysis_start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            analysis_end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
        )

        summary = generate_health_summary(metrics)

        assert "⚠️ Warning" in summary
        assert "35.00%" in summary
        assert "acquisition dependence" in summary

    def test_generate_retention_insights(self):
        """Retention insights should analyze migration patterns."""
        p1 = Lens1Metrics(
            100,
            40,
            Decimal("40.00"),
            Decimal("10000.00"),
            Decimal("45.00"),
            Decimal("62.00"),
            Decimal("2.50"),
            Decimal("100.00"),
            {},
        )
        p2 = Lens1Metrics(
            120,
            50,
            Decimal("41.67"),
            Decimal("12000.00"),
            Decimal("47.00"),
            Decimal("64.00"),
            Decimal("2.80"),
            Decimal("100.00"),
            {},
        )
        migration = CustomerMigration(
            retained=frozenset(["C1", "C2"]),
            churned=frozenset(["C3"]),
            new=frozenset(["C4", "C5", "C6"]),
            reactivated=frozenset(["C5"]),
        )
        metrics = Lens2Metrics(
            period1_metrics=p1,
            period2_metrics=p2,
            migration=migration,
            retention_rate=Decimal("66.67"),
            churn_rate=Decimal("33.33"),
            reactivation_rate=Decimal("33.33"),
            customer_count_change=20,
            revenue_change_pct=Decimal("20.00"),
            avg_order_value_change_pct=Decimal("5.00"),
        )

        insights = generate_retention_insights(metrics)

        assert "Retention & Churn" in insights
        assert "66.67%" in insights
        assert "Strategic Recommendations" in insights

    def test_generate_cohort_comparison(self):
        """Cohort comparison should analyze performance trends."""
        decomp = CohortDecomposition(
            cohort_id="2023-Q1",
            period_number=0,
            cohort_size=100,
            active_customers=100,
            pct_active=Decimal("100.00"),
            total_orders=150,
            aof=Decimal("1.50"),
            total_revenue=Decimal("15000.00"),
            aov=Decimal("100.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("15000.00"),
        )
        ttsp = TimeToSecondPurchase(
            cohort_id="2023-Q1",
            customers_with_repeat=60,
            repeat_rate=Decimal("60.00"),
            median_days=Decimal("30.00"),
            mean_days=Decimal("35.00"),
            cumulative_repeat_by_period={},
        )
        metrics = Lens4Metrics(
            cohort_decompositions=[decomp],
            time_to_second_purchase=[ttsp],
            cohort_comparisons=[],
            alignment_type="left-aligned",
        )

        summary = generate_cohort_comparison(metrics)

        assert "Multi-Cohort Performance" in summary
        assert "2023-Q1" in summary
        assert "60.00%" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_rfm_distribution(self):
        """Empty RFM distribution should not break formatting."""
        metrics = Lens1Metrics(
            total_customers=100,
            one_time_buyers=30,
            one_time_buyer_pct=Decimal("30.00"),
            total_revenue=Decimal("10000.00"),
            top_10pct_revenue_contribution=Decimal("45.00"),
            top_20pct_revenue_contribution=Decimal("62.00"),
            avg_orders_per_customer=Decimal("2.50"),
            median_customer_value=Decimal("100.00"),
            rfm_distribution={},
        )

        table = format_lens1_table(metrics)
        assert table  # Should not crash

    def test_zero_customers_lens1(self):
        """Zero customers should be handled gracefully."""
        metrics = Lens1Metrics(
            total_customers=0,
            one_time_buyers=0,
            one_time_buyer_pct=Decimal("0.00"),
            total_revenue=Decimal("0.00"),
            top_10pct_revenue_contribution=Decimal("0.00"),
            top_20pct_revenue_contribution=Decimal("0.00"),
            avg_orders_per_customer=Decimal("0.00"),
            median_customer_value=Decimal("0.00"),
            rfm_distribution={},
        )

        table = format_lens1_table(metrics)
        assert "0" in table

    def test_single_period_lens3(self):
        """Single period Lens3 should work."""
        metrics = Lens3Metrics(
            cohort_name="2023-Q1",
            acquisition_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
            cohort_size=100,
            periods=[
                CohortPeriodMetrics(0, 100, 1.0, 1.5, 50.0, 1.5, 50.0, 5000.0),
            ],
        )

        result = create_retention_trend_chart(metrics)
        assert "plotly_json" in result
        assert len(result["plotly_json"]["data"]) == 2


class TestFormatConsistency:
    """Test that all formatters produce valid output."""

    def test_markdown_tables_are_valid_markdown(self):
        """All markdown tables should have proper table structure."""
        # Create sample metrics
        lens1 = Lens1Metrics(
            100,
            30,
            Decimal("30.00"),
            Decimal("10000.00"),
            Decimal("45.00"),
            Decimal("62.00"),
            Decimal("2.50"),
            Decimal("100.00"),
            {},
        )

        table = format_lens1_table(lens1)

        # Check for markdown table markers
        assert "|" in table
        assert "---" in table or "##" in table

    def test_plotly_charts_have_required_fields(self):
        """All Plotly charts should have data and layout."""
        lens1 = Lens1Metrics(
            100,
            30,
            Decimal("30.00"),
            Decimal("10000.00"),
            Decimal("45.00"),
            Decimal("62.00"),
            Decimal("2.50"),
            Decimal("100.00"),
            {},
        )

        result = create_revenue_concentration_pie(lens1)

        # Verify dual format
        assert "plotly_json" in result
        assert "image_base64" in result
        chart = result["plotly_json"]
        assert "data" in chart
        assert "layout" in chart
        assert isinstance(chart["data"], list)
        assert isinstance(chart["layout"], dict)

    def test_summaries_are_non_empty(self):
        """All summaries should produce non-empty strings."""
        health = CustomerBaseHealthScore(
            1000,
            800,
            Decimal("80.00"),
            "improving",
            Decimal("85.00"),
            Decimal("15.00"),
            Decimal("82.50"),
            "B",
        )
        metrics = Lens5Metrics(
            [],
            [],
            health,
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 12, 31, tzinfo=timezone.utc),
        )

        summary = generate_health_summary(metrics)

        assert len(summary) > 100  # Should be substantive
        assert "#" in summary  # Should have markdown headers


class TestChartOptimization:
    """Test chart size optimization features (Issue #129)."""

    def test_chart_config_defaults(self):
        """Chart config should default to medium quality (800x400)."""
        from customer_base_audit.mcp.formatters import ChartConfig

        config = ChartConfig()
        assert config.width == 800
        assert config.height == 400
        assert config.quality == "medium"

    def test_chart_config_quality_presets(self):
        """Quality presets should set appropriate dimensions."""
        from customer_base_audit.mcp.formatters import ChartConfig

        high = ChartConfig.from_quality("high")
        assert high.width == 1200
        assert high.height == 600

        medium = ChartConfig.from_quality("medium")
        assert medium.width == 800
        assert medium.height == 400

        low = ChartConfig.from_quality("low")
        assert low.width == 600
        assert low.height == 300

    def test_charts_use_config(self):
        """Charts should respect global config settings."""
        from customer_base_audit.mcp.formatters import (
            ChartConfig,
            get_chart_config,
            set_chart_config,
        )

        # Save original config
        original_config = get_chart_config()

        try:
            # Set to low quality
            set_chart_config(ChartConfig.from_quality("low"))

            # Create chart and verify it uses smaller dimensions
            health = CustomerBaseHealthScore(
                1000,
                800,
                Decimal("80.00"),
                "improving",
                Decimal("85.00"),
                Decimal("15.00"),
                Decimal("82.50"),
                "B",
            )
            metrics = Lens5Metrics(
                [],
                [],
                health,
                datetime(2023, 1, 1, tzinfo=timezone.utc),
                datetime(2023, 12, 31, tzinfo=timezone.utc),
            )

            result = create_health_score_gauge(metrics)

            # Should use low quality height (300px)
            assert result["plotly_json"]["layout"]["height"] == 300

            # Now test high quality
            set_chart_config(ChartConfig.from_quality("high"))
            result = create_health_score_gauge(metrics)
            assert result["plotly_json"]["layout"]["height"] == 600

        finally:
            # Restore original config
            set_chart_config(original_config)

    def test_chart_size_reduces_token_usage(self):
        """Smaller charts should produce smaller JSON output."""
        from customer_base_audit.mcp.formatters import ChartConfig, set_chart_config
        import json

        # Save original config
        from customer_base_audit.mcp.formatters import get_chart_config

        original_config = get_chart_config()

        try:
            lens1 = Lens1Metrics(
                100,
                30,
                Decimal("30.00"),
                Decimal("10000.00"),
                Decimal("45.00"),
                Decimal("62.00"),
                Decimal("2.50"),
                Decimal("100.00"),
                {},
            )

            # Generate chart at high quality
            set_chart_config(ChartConfig.from_quality("high"))
            high_result = create_revenue_concentration_pie(lens1)
            high_size = len(json.dumps(high_result))

            # Generate chart at low quality
            set_chart_config(ChartConfig.from_quality("low"))
            low_result = create_revenue_concentration_pie(lens1)
            low_size = len(json.dumps(low_result))

            # Low quality should be smaller (though for simple charts difference is minimal)
            # The real savings come from not embedding large images
            assert low_size <= high_size

        finally:
            set_chart_config(original_config)
