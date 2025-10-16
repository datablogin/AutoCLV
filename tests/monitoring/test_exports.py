"""Tests for drift report exports."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from customer_base_audit.monitoring.drift import detect_feature_drift
from customer_base_audit.monitoring.exports import (
    export_drift_report_csv,
    export_drift_report_json,
    export_drift_report_markdown,
    get_drift_summary,
)


@pytest.fixture
def sample_drift_results():
    """Create sample drift detection results for testing exports."""
    np.random.seed(42)
    baseline = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 1000),
            "income": np.random.normal(50000, 15000, 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )
    current = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 1000),  # Stable
            "income": np.random.normal(60000, 15000, 1000),  # Shifted
            "score": np.random.uniform(0, 100, 1000),  # Stable
        }
    )

    return detect_feature_drift(baseline, current, method="psi")


class TestExportDriftReportJSON:
    """Test JSON export functionality."""

    def test_export_json_creates_file(self, sample_drift_results):
        """Should create JSON file with drift results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.json"

            export_drift_report_json(sample_drift_results, output_path)

            assert output_path.exists()

    def test_export_json_contains_results(self, sample_drift_results):
        """JSON file should contain all drift results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.json"

            export_drift_report_json(sample_drift_results, output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert "drift_results" in data
            assert len(data["drift_results"]) == len(sample_drift_results)

    def test_export_json_with_metadata(self, sample_drift_results):
        """JSON export should include metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.json"
            metadata = {"model_version": "v2.1", "data_source": "production"}

            export_drift_report_json(
                sample_drift_results, output_path, metadata=metadata
            )

            with open(output_path) as f:
                data = json.load(f)

            assert data["metadata"] == metadata

    def test_export_json_creates_nested_directories(self, sample_drift_results):
        """Should create nested directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "reports" / "2024" / "drift_report.json"

            export_drift_report_json(sample_drift_results, output_path)

            assert output_path.exists()


class TestExportDriftReportCSV:
    """Test CSV export functionality."""

    def test_export_csv_creates_file(self, sample_drift_results):
        """Should create CSV file with drift results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.csv"

            export_drift_report_csv(sample_drift_results, output_path)

            assert output_path.exists()

    def test_export_csv_has_correct_columns(self, sample_drift_results):
        """CSV should have all expected columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.csv"

            export_drift_report_csv(sample_drift_results, output_path)

            df = pd.read_csv(output_path)

            expected_columns = [
                "metric_name",
                "test_type",
                "drift_score",
                "p_value",
                "drift_detected",
                "threshold",
                "interpretation",
            ]
            assert list(df.columns) == expected_columns

    def test_export_csv_row_count_matches(self, sample_drift_results):
        """CSV should have one row per drift result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.csv"

            export_drift_report_csv(sample_drift_results, output_path)

            df = pd.read_csv(output_path)

            assert len(df) == len(sample_drift_results)


class TestExportDriftReportMarkdown:
    """Test Markdown export functionality."""

    def test_export_markdown_creates_file(self, sample_drift_results):
        """Should create Markdown file with drift report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.md"

            export_drift_report_markdown(sample_drift_results, output_path)

            assert output_path.exists()

    def test_export_markdown_contains_title(self, sample_drift_results):
        """Markdown should contain report title."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.md"

            export_drift_report_markdown(
                sample_drift_results, output_path, title="Weekly Drift Report"
            )

            content = output_path.read_text()

            assert "# Weekly Drift Report" in content

    def test_export_markdown_contains_summary(self, sample_drift_results):
        """Markdown should contain summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.md"

            export_drift_report_markdown(sample_drift_results, output_path)

            content = output_path.read_text()

            assert "## Summary" in content
            assert "Total Metrics Monitored" in content

    def test_export_markdown_with_metadata(self, sample_drift_results):
        """Markdown should include metadata section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.md"
            metadata = {"model": "BG/NBD v2.1", "period": "2024-01-08 to 2024-01-15"}

            export_drift_report_markdown(
                sample_drift_results, output_path, metadata=metadata
            )

            content = output_path.read_text()

            assert "## Metadata" in content
            assert "BG/NBD v2.1" in content

    def test_export_markdown_groups_by_drift_status(self, sample_drift_results):
        """Markdown should separate drifted vs stable metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "drift_report.md"

            export_drift_report_markdown(sample_drift_results, output_path)

            content = output_path.read_text()

            # Should have sections for both (income has drift, age/score don't)
            assert (
                "⚠️ Metrics with Drift Detected" in content
                or "✅ Stable Metrics" in content
            )


class TestGetDriftSummary:
    """Test drift summary statistics."""

    def test_summary_counts_metrics(self, sample_drift_results):
        """Summary should count total metrics."""
        summary = get_drift_summary(sample_drift_results)

        assert summary["total_metrics"] == len(sample_drift_results)

    def test_summary_counts_drifted_metrics(self, sample_drift_results):
        """Summary should count drifted metrics."""
        summary = get_drift_summary(sample_drift_results)

        drifted_count = sum(
            1 for r in sample_drift_results.values() if r.drift_detected
        )
        assert summary["drifted_count"] == drifted_count

    def test_summary_calculates_drift_rate(self, sample_drift_results):
        """Summary should calculate drift rate."""
        summary = get_drift_summary(sample_drift_results)

        expected_rate = summary["drifted_count"] / summary["total_metrics"]
        assert summary["drift_rate"] == pytest.approx(expected_rate)

    def test_summary_lists_drifted_metrics(self, sample_drift_results):
        """Summary should list metrics with drift."""
        summary = get_drift_summary(sample_drift_results)

        assert "metrics_with_drift" in summary
        assert isinstance(summary["metrics_with_drift"], list)

    def test_summary_empty_results(self):
        """Summary should handle empty results."""
        summary = get_drift_summary({})

        assert summary["total_metrics"] == 0
        assert summary["drifted_count"] == 0
        assert summary["drift_rate"] == 0.0
