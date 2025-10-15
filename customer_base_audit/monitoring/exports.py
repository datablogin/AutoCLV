"""Export drift detection results to various formats.

This module provides utilities for saving and reporting drift detection results
for model monitoring dashboards, alerts, and audit trails.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from customer_base_audit.monitoring.drift import DriftResult

logger = logging.getLogger(__name__)


def export_drift_report_json(
    results: dict[str, DriftResult],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Export drift detection results to JSON format.

    Creates a structured JSON report suitable for dashboards, alerting systems,
    or long-term storage.

    Parameters
    ----------
    results:
        Dictionary mapping metric names to DriftResult objects
    output_path:
        Path where JSON file will be saved
    metadata:
        Optional metadata to include in report (e.g., model version, timestamp)

    Examples
    --------
    >>> results = detect_feature_drift(baseline_df, current_df)
    >>> export_drift_report_json(
    ...     results,
    ...     "drift_report_2024-01-15.json",
    ...     metadata={"model_version": "v2.1", "data_source": "production"}
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert DriftResult objects to dicts
    report_data = {
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat(),
        "drift_results": {},
    }

    for key, result in results.items():
        report_data["drift_results"][key] = {
            "metric_name": result.metric_name,
            "test_type": result.test_type,
            "drift_score": float(result.drift_score),
            "p_value": float(result.p_value) if result.p_value is not None else None,
            "drift_detected": bool(
                result.drift_detected
            ),  # Convert numpy bool to Python bool
            "threshold": float(result.threshold),
            "interpretation": result.interpretation,
        }

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Drift report exported to {output_path}")


def export_drift_report_csv(
    results: dict[str, DriftResult],
    output_path: str | Path,
) -> None:
    """Export drift detection results to CSV format.

    Creates a tabular CSV suitable for spreadsheet analysis or database import.

    Parameters
    ----------
    results:
        Dictionary mapping metric names to DriftResult objects
    output_path:
        Path where CSV file will be saved

    Examples
    --------
    >>> results = detect_feature_drift(baseline_df, current_df)
    >>> export_drift_report_csv(results, "drift_report_2024-01-15.csv")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to DataFrame
    records = []
    for key, result in results.items():
        records.append(
            {
                "metric_name": result.metric_name,
                "test_type": result.test_type,
                "drift_score": result.drift_score,
                "p_value": result.p_value,
                "drift_detected": result.drift_detected,
                "threshold": result.threshold,
                "interpretation": result.interpretation,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    logger.info(f"Drift report exported to {output_path}")


def export_drift_report_markdown(
    results: dict[str, DriftResult],
    output_path: str | Path,
    title: str = "Drift Detection Report",
    metadata: dict[str, Any] | None = None,
) -> None:
    """Export drift detection results to Markdown format.

    Creates a human-readable Markdown report suitable for documentation or
    stakeholder communication.

    Parameters
    ----------
    results:
        Dictionary mapping metric names to DriftResult objects
    output_path:
        Path where Markdown file will be saved
    title:
        Report title (default: "Drift Detection Report")
    metadata:
        Optional metadata to include in report header

    Examples
    --------
    >>> results = detect_feature_drift(baseline_df, current_df)
    >>> export_drift_report_markdown(
    ...     results,
    ...     "drift_report.md",
    ...     title="Weekly Drift Report - 2024-01-15",
    ...     metadata={"model": "BG/NBD v2.1", "period": "2024-01-08 to 2024-01-15"}
    ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# {title}\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if metadata:
        lines.append("## Metadata\n")
        for key, value in metadata.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    lines.append("## Summary\n")
    total_metrics = len(results)
    drifted_metrics = sum(1 for r in results.values() if r.drift_detected)
    lines.append(f"- **Total Metrics Monitored:** {total_metrics}")
    lines.append(f"- **Metrics with Drift:** {drifted_metrics}")
    lines.append(
        f"- **Drift Rate:** {100 * drifted_metrics / total_metrics if total_metrics > 0 else 0:.1f}%\n"
    )

    # Group by drift status
    drifted = {k: v for k, v in results.items() if v.drift_detected}
    stable = {k: v for k, v in results.items() if not v.drift_detected}

    if drifted:
        lines.append("## ⚠️ Metrics with Drift Detected\n")
        lines.append("| Metric | Test | Score | Threshold | Interpretation |")
        lines.append("|--------|------|-------|-----------|----------------|")
        for key, result in sorted(drifted.items()):
            score_str = (
                f"{result.drift_score:.4f}"
                if result.p_value is None
                else f"{result.drift_score:.4f} (p={result.p_value:.4f})"
            )
            lines.append(
                f"| {result.metric_name} | {result.test_type.upper()} | {score_str} | {result.threshold:.4f} | {result.interpretation} |"
            )
        lines.append("")

    if stable:
        lines.append("## ✅ Stable Metrics (No Drift)\n")
        lines.append("| Metric | Test | Score | Threshold |")
        lines.append("|--------|------|-------|-----------|")
        for key, result in sorted(stable.items()):
            score_str = (
                f"{result.drift_score:.4f}"
                if result.p_value is None
                else f"{result.drift_score:.4f} (p={result.p_value:.4f})"
            )
            lines.append(
                f"| {result.metric_name} | {result.test_type.upper()} | {score_str} | {result.threshold:.4f} |"
            )
        lines.append("")

    lines.append("## Interpretation Guide\n")
    lines.append("### PSI (Population Stability Index)\n")
    lines.append("- **< 0.1:** No significant drift")
    lines.append("- **0.1 - 0.25:** Small drift, investigation recommended")
    lines.append("- **≥ 0.25:** Significant drift, retraining recommended\n")
    lines.append("### KS (Kolmogorov-Smirnov Test)\n")
    lines.append(
        "- **p-value < 0.05:** Distributions differ significantly (drift detected)"
    )
    lines.append("- **p-value ≥ 0.05:** Distributions are similar (no drift)")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Drift report exported to {output_path}")


def get_drift_summary(results: dict[str, DriftResult]) -> dict[str, Any]:
    """Get a summary of drift detection results.

    Returns a dictionary with key statistics suitable for dashboards or alerting.

    Parameters
    ----------
    results:
        Dictionary mapping metric names to DriftResult objects

    Returns
    -------
    dict[str, Any]
        Summary statistics including total metrics, drifted count, and severities

    Examples
    --------
    >>> results = detect_feature_drift(baseline_df, current_df)
    >>> summary = get_drift_summary(results)
    >>> print(f"Drift detected in {summary['drifted_count']} of {summary['total_metrics']} metrics")
    """
    if not results:
        return {
            "total_metrics": 0,
            "drifted_count": 0,
            "drift_rate": 0.0,
            "severity_counts": {"low": 0, "medium": 0, "high": 0},
        }

    total_metrics = len(results)
    drifted_count = sum(1 for r in results.values() if r.drift_detected)

    # Categorize severity for both PSI and KS tests
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    for result in results.values():
        if result.drift_detected:
            if result.test_type == "psi":
                # PSI severity based on drift score
                if result.drift_score < 0.1:
                    severity_counts["low"] += 1
                elif result.drift_score < 0.25:
                    severity_counts["medium"] += 1
                else:
                    severity_counts["high"] += 1
            elif result.test_type == "ks" and result.p_value is not None:
                # KS severity based on p-value (lower p-value = higher severity)
                if result.p_value < 0.001:
                    severity_counts["high"] += 1  # Very strong evidence of drift
                elif result.p_value < 0.01:
                    severity_counts["medium"] += 1  # Strong evidence of drift
                else:
                    severity_counts["low"] += 1  # Moderate evidence of drift (p < 0.05)

    return {
        "total_metrics": total_metrics,
        "drifted_count": drifted_count,
        "drift_rate": drifted_count / total_metrics if total_metrics > 0 else 0.0,
        "severity_counts": severity_counts,
        "metrics_with_drift": [
            result.metric_name for result in results.values() if result.drift_detected
        ],
    }
