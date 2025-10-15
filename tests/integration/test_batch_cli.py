"""Integration tests for CLI batch processing commands.

Tests the complete workflow from raw transaction data through CLI commands
to final output files (CLV scores CSV and Five Lenses Markdown reports).
"""

import pandas as pd
import pytest

from customer_base_audit.cli import (
    generate_five_lenses_report_cli,
    score_clv_cli,
)


@pytest.fixture
def sample_transactions_json(tmp_path):
    """Create sample transactions JSON file for testing."""
    import json

    # Generate realistic transaction data spanning multiple periods
    transactions = [
        # Customer C1: High-value repeat buyer
        {
            "order_id": "O1",
            "customer_id": "C1",
            "event_ts": "2023-01-15T10:00:00+00:00",
            "unit_price": 100.0,
            "quantity": 2,
        },
        {
            "order_id": "O2",
            "customer_id": "C1",
            "event_ts": "2023-02-20T14:30:00+00:00",
            "unit_price": 150.0,
            "quantity": 1,
        },
        {
            "order_id": "O3",
            "customer_id": "C1",
            "event_ts": "2023-04-10T09:15:00+00:00",
            "unit_price": 120.0,
            "quantity": 3,
        },
        # Customer C2: One-time buyer
        {
            "order_id": "O4",
            "customer_id": "C2",
            "event_ts": "2023-01-25T16:45:00+00:00",
            "unit_price": 50.0,
            "quantity": 1,
        },
        # Customer C3: Moderate repeat buyer
        {
            "order_id": "O5",
            "customer_id": "C3",
            "event_ts": "2023-02-05T11:20:00+00:00",
            "unit_price": 80.0,
            "quantity": 2,
        },
        {
            "order_id": "O6",
            "customer_id": "C3",
            "event_ts": "2023-05-12T13:00:00+00:00",
            "unit_price": 90.0,
            "quantity": 1,
        },
        # Customer C4: New in Q2
        {
            "order_id": "O7",
            "customer_id": "C4",
            "event_ts": "2023-04-20T15:30:00+00:00",
            "unit_price": 200.0,
            "quantity": 1,
        },
        # Customer C5: New in Q2, high-value repeat
        {
            "order_id": "O8",
            "customer_id": "C5",
            "event_ts": "2023-06-01T10:00:00+00:00",
            "unit_price": 300.0,
            "quantity": 2,
        },
        {
            "order_id": "O9",
            "customer_id": "C5",
            "event_ts": "2023-06-15T14:00:00+00:00",
            "unit_price": 250.0,
            "quantity": 1,
        },
    ]

    json_path = tmp_path / "transactions.json"
    with open(json_path, "w") as f:
        json.dump(transactions, f, indent=2)

    return json_path


class TestScoreCLVCLI:
    """Test score_clv CLI command."""

    def test_score_clv_basic_workflow(self, sample_transactions_json, tmp_path):
        """Test basic CLV scoring workflow."""
        output_csv = tmp_path / "clv_scores.csv"

        # Run CLI command
        exit_code = score_clv_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_csv),
                "--time-horizon",
                "12",
                "--profit-margin",
                "0.30",
                "--discount-rate",
                "0.10",
            ]
        )

        assert exit_code == 0
        assert output_csv.exists()

        # Verify output CSV structure
        df = pd.read_csv(output_csv)

        expected_columns = [
            "customer_id",
            "predicted_purchases",
            "predicted_avg_value",
            "clv",
            "prob_alive",
        ]
        assert list(df.columns) == expected_columns

        # Verify we have scores for all customers
        assert len(df) == 5  # C1, C2, C3, C4, C5

        # Verify all customers have non-negative CLV
        assert (df["clv"] >= 0).all()

        # Verify prob_alive is in [0, 1]
        assert (df["prob_alive"] >= 0).all()
        assert (df["prob_alive"] <= 1).all()

    def test_score_clv_custom_parameters(self, sample_transactions_json, tmp_path):
        """Test CLV scoring with custom parameters."""
        output_csv = tmp_path / "clv_custom.csv"

        exit_code = score_clv_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_csv),
                "--time-horizon",
                "6",
                "--profit-margin",
                "0.20",
                "--discount-rate",
                "0.05",
                "--min-frequency",
                "2",
            ]
        )

        assert exit_code == 0
        assert output_csv.exists()

    @pytest.mark.skip(
        reason="Known limitation: custom observation dates require periods to be pre-filtered "
        "to match observation window. See Issue #36 follow-up."
    )
    def test_score_clv_with_observation_dates(self, sample_transactions_json, tmp_path):
        """Test CLV scoring with specified observation period."""
        output_csv = tmp_path / "clv_dates.csv"

        exit_code = score_clv_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_csv),
                "--observation-start",
                "2023-01-02",
                "--observation-end",
                "2023-06-29",
            ]
        )

        assert exit_code == 0
        assert output_csv.exists()

    def test_score_clv_output_values_reasonable(
        self, sample_transactions_json, tmp_path
    ):
        """Test that CLV output values are reasonable."""
        output_csv = tmp_path / "clv_values.csv"

        score_clv_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_csv),
            ]
        )

        df = pd.read_csv(output_csv)

        # CLV values should be positive for at least some customers
        assert df["clv"].sum() > 0

        # High-value customers (C1, C5) should have higher CLV
        c1_clv = df[df["customer_id"] == "C1"]["clv"].values[0]
        c2_clv = df[df["customer_id"] == "C2"]["clv"].values[0]

        # C1 (repeat buyer) should have higher CLV than C2 (one-time buyer)
        assert c1_clv > c2_clv

    def test_score_clv_empty_transactions(self, tmp_path):
        """Test CLV scoring with empty transactions file."""
        import json

        empty_json = tmp_path / "empty.json"
        with open(empty_json, "w") as f:
            json.dump([], f)

        output_csv = tmp_path / "clv_empty.csv"

        exit_code = score_clv_cli(
            [
                str(empty_json),
                "--output",
                str(output_csv),
            ]
        )

        # Should fail gracefully with non-zero exit code
        assert exit_code != 0


class TestGenerateFiveLensesReportCLI:
    """Test generate_five_lenses_report CLI command."""

    def test_five_lenses_basic_workflow(self, sample_transactions_json, tmp_path):
        """Test basic Five Lenses report generation."""
        output_md = tmp_path / "five_lenses_report.md"

        exit_code = generate_five_lenses_report_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_md),
                "--period-granularity",
                "quarter",
            ]
        )

        assert exit_code == 0
        assert output_md.exists()

        # Verify report contains expected sections
        content = output_md.read_text()

        assert "# Five Lenses Customer Base Audit Report" in content
        assert "## Lens 1: Single Period Analysis" in content
        assert "## Lens 2: Period-to-Period Comparison" in content
        assert "## Lens 3: Cohort Evolution" in content
        assert "## Lens 5: Overall Customer Base Health" in content

        # Verify key metrics are present
        assert "Total Customers:" in content
        assert "Total Revenue:" in content
        assert "Retention Rate:" in content
        assert "Health Score:" in content
        assert "Health Grade:" in content

    def test_five_lenses_monthly_granularity(self, sample_transactions_json, tmp_path):
        """Test Five Lenses report with monthly granularity."""
        output_md = tmp_path / "five_lenses_monthly.md"

        exit_code = generate_five_lenses_report_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_md),
                "--period-granularity",
                "month",
            ]
        )

        assert exit_code == 0
        assert output_md.exists()

        content = output_md.read_text()
        assert "**Analysis Granularity:** month" in content

    def test_five_lenses_with_observation_end(self, sample_transactions_json, tmp_path):
        """Test Five Lenses report with specified observation end."""
        output_md = tmp_path / "five_lenses_date.md"

        exit_code = generate_five_lenses_report_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_md),
                "--observation-end",
                "2023-06-30",
            ]
        )

        assert exit_code == 0
        assert output_md.exists()

    def test_five_lenses_cohort_analysis(self, sample_transactions_json, tmp_path):
        """Test that Five Lenses report includes cohort analysis."""
        output_md = tmp_path / "five_lenses_cohorts.md"

        generate_five_lenses_report_cli(
            [
                str(sample_transactions_json),
                "--output",
                str(output_md),
            ]
        )

        content = output_md.read_text()

        # Should have multiple cohorts
        assert "cohort_" in content.lower()

        # Should have cohort size info
        assert "Cohort Size:" in content


class TestCLIBatchProcessing1000Customers:
    """Test CLI commands with 1,000+ customers (Issue #36 acceptance criterion)."""

    @pytest.fixture
    def large_dataset_json(self, tmp_path):
        """Generate dataset with 1,000+ customers."""
        import json
        import random

        random.seed(42)

        transactions = []
        order_id_counter = 1

        # Generate 1,200 customers
        for customer_id in range(1, 1201):
            cust_id = f"C{customer_id}"

            # Random number of purchases (1-10)
            num_purchases = random.randint(1, 10)

            for purchase_num in range(num_purchases):
                # Random date in 2023
                month = random.randint(1, 6)
                day = random.randint(1, 28)
                hour = random.randint(0, 23)

                transaction = {
                    "order_id": f"O{order_id_counter}",
                    "customer_id": cust_id,
                    "event_ts": f"2023-{month:02d}-{day:02d}T{hour:02d}:00:00+00:00",
                    "unit_price": round(random.uniform(10, 500), 2),
                    "quantity": random.randint(1, 5),
                }

                transactions.append(transaction)
                order_id_counter += 1

        json_path = tmp_path / "large_dataset.json"
        with open(json_path, "w") as f:
            json.dump(transactions, f)

        return json_path

    def test_score_clv_with_1000_customers(self, large_dataset_json, tmp_path):
        """Test CLV scoring processes 1,000+ customers successfully."""
        output_csv = tmp_path / "clv_1000.csv"

        exit_code = score_clv_cli(
            [
                str(large_dataset_json),
                "--output",
                str(output_csv),
            ]
        )

        assert exit_code == 0
        assert output_csv.exists()

        # Verify output has correct number of customers
        df = pd.read_csv(output_csv)
        assert len(df) == 1200

        # Verify output schema
        expected_columns = [
            "customer_id",
            "predicted_purchases",
            "predicted_avg_value",
            "clv",
            "prob_alive",
        ]
        assert list(df.columns) == expected_columns

        # Verify data types
        assert pd.api.types.is_numeric_dtype(df["predicted_purchases"])
        assert pd.api.types.is_numeric_dtype(df["predicted_avg_value"])
        assert pd.api.types.is_numeric_dtype(df["clv"])
        assert pd.api.types.is_numeric_dtype(df["prob_alive"])

    def test_five_lenses_with_1000_customers(self, large_dataset_json, tmp_path):
        """Test Five Lenses report generation with 1,000+ customers."""
        output_md = tmp_path / "five_lenses_1000.md"

        exit_code = generate_five_lenses_report_cli(
            [
                str(large_dataset_json),
                "--output",
                str(output_md),
            ]
        )

        assert exit_code == 0
        assert output_md.exists()

        # Verify report contains expected content
        content = output_md.read_text()

        # Should have all lens sections
        assert "## Lens 1:" in content
        assert "## Lens 2:" in content
        assert "## Lens 3:" in content
        assert "## Lens 5:" in content

        # Should have customer count >= 1000
        # Extract total customers from Lens 5 (which has total across all cohorts)
        import re

        match = re.search(r"\*\*Total Customers:\*\*\s*(\d+)", content)
        assert match is not None
        total_customers = int(match.group(1))
        assert total_customers >= 1000
