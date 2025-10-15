"""Command line entry points for the AutoCLV toolkit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from customer_base_audit.analyses.lens1 import analyze_single_period
from customer_base_audit.analyses.lens2 import analyze_period_comparison
from customer_base_audit.analyses.lens3 import analyze_cohort_evolution
from customer_base_audit.analyses.lens5 import assess_customer_base_health
from customer_base_audit.foundation import (
    CustomerDataMartBuilder,
    PeriodGranularity,
)
from customer_base_audit.foundation.cohorts import CohortDefinition, assign_cohorts
from customer_base_audit.foundation.customer_contract import CustomerIdentifier
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper
from customer_base_audit.models.clv_calculator import CLVCalculator
from customer_base_audit.models.gamma_gamma import (
    GammaGammaConfig,
    GammaGammaModelWrapper,
)
from customer_base_audit.models.model_prep import (
    prepare_bg_nbd_inputs,
    prepare_gamma_gamma_inputs,
)

logger = logging.getLogger(__name__)


MAX_INPUT_BYTES = 25 * 1024 * 1024  # 25 MiB cap to avoid accidental OOM


def _load_transactions(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve()
    size = resolved.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise ValueError(
            f"Input file {resolved} is {size} bytes; exceeds limit of {MAX_INPUT_BYTES} bytes"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):  # pragma: no cover - defensive
        raise ValueError("Expected a list of transactions in the input file")

    # Convert timestamp strings to datetime objects
    transactions = []
    for item in payload:
        txn = dict(item)
        # Handle event_ts or order_ts
        ts_field = "event_ts" if "event_ts" in txn else "order_ts"
        if ts_field in txn and isinstance(txn[ts_field], str):
            txn[ts_field] = datetime.fromisoformat(txn[ts_field].replace("Z", "+00:00"))
        transactions.append(txn)

    return transactions


def build_data_mart_cli(argv: list[str] | None = None) -> int:
    """Build order and customer-period aggregations from a JSON file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Path to JSON file with raw transactions"
    )
    parser.add_argument(
        "--period",
        dest="periods",
        action="append",
        choices=[item.value for item in PeriodGranularity],
        help="Period granularities to compute (defaults to quarter and year).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for writing the aggregated data mart as JSON.",
    )

    args = parser.parse_args(argv)
    periods = (
        tuple(PeriodGranularity(period) for period in args.periods)
        if args.periods
        else None
    )
    builder = CustomerDataMartBuilder(period_granularities=periods)
    transactions = _load_transactions(args.input)
    mart = builder.build(transactions)

    if args.output:
        output_path = args.output.resolve()
        cwd = Path.cwd().resolve()
        try:
            output_path.relative_to(cwd)
        except ValueError:
            raise ValueError(
                f"Output path {output_path} must reside within the current working directory"
            )
        payload = mart.as_dict()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    else:  # stdout fallback enables piping in shell usage.
        json.dump(mart.as_dict(), fp=sys.stdout, indent=2, sort_keys=True)
        print()

    return 0


def score_clv_cli(argv: list[str] | None = None) -> int:
    """Calculate CLV scores from transaction data and export to CSV.

    This command processes transaction data through the complete CLV pipeline:
    1. Builds customer data mart from transactions
    2. Prepares inputs for BG/NBD (purchase frequency) and Gamma-Gamma (monetary value) models
    3. Fits probabilistic models using MAP estimation
    4. Combines predictions to calculate customer lifetime value scores
    5. Exports results to CSV with columns: customer_id, predicted_purchases,
       predicted_avg_value, clv, prob_alive

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Calculate CLV scores from transaction data"
    )
    parser.add_argument(
        "input", type=Path, help="Path to JSON file with raw transactions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for output CSV file with CLV scores",
    )
    parser.add_argument(
        "--observation-start",
        type=str,
        help="Observation start date (ISO format: YYYY-MM-DD). Defaults to first transaction date.",
    )
    parser.add_argument(
        "--observation-end",
        type=str,
        help="Observation end date (ISO format: YYYY-MM-DD). Defaults to last transaction date.",
    )
    parser.add_argument(
        "--time-horizon",
        type=int,
        default=12,
        help="Prediction time horizon in months (default: 12)",
    )
    parser.add_argument(
        "--profit-margin",
        type=float,
        default=0.30,
        help="Profit margin as decimal (default: 0.30 = 30%%)",
    )
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=0.10,
        help="Annual discount rate for present value (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum transaction frequency for Gamma-Gamma model (default: 2)",
    )

    args = parser.parse_args(argv)

    #  Load transactions
    logger.info(f"Loading transactions from {args.input}")
    transactions = _load_transactions(args.input)

    if not transactions:
        logger.error("No transactions found in input file")
        return 1

    # Build data mart
    logger.info(f"Building data mart from {len(transactions)} transactions")
    builder = CustomerDataMartBuilder()
    mart = builder.build(transactions)

    # Determine observation period
    # Use period boundaries rather than transaction times for proper alignment
    all_periods = []
    for granularity_periods in mart.periods.values():
        all_periods.extend(granularity_periods)

    if args.observation_start:
        observation_start = datetime.fromisoformat(args.observation_start).replace(
            tzinfo=timezone.utc
        )
    else:
        observation_start = (
            min(p.period_start for p in all_periods)
            if all_periods
            else min(order.order_ts for order in mart.orders)
        )

    if args.observation_end:
        observation_end = datetime.fromisoformat(args.observation_end).replace(
            tzinfo=timezone.utc
        )
    else:
        observation_end = (
            max(p.period_end for p in all_periods)
            if all_periods
            else max(order.order_ts for order in mart.orders)
        )

    logger.info(
        f"Observation period: {observation_start.date()} to {observation_end.date()}"
    )

    if not all_periods:
        logger.error("No period aggregations generated from transactions")
        return 1

    # Prepare BG/NBD inputs
    logger.info("Preparing BG/NBD model inputs")
    bg_nbd_data = prepare_bg_nbd_inputs(all_periods, observation_start, observation_end)

    if bg_nbd_data.empty:
        logger.error("No customers found for BG/NBD model")
        return 1

    logger.info(f"Prepared BG/NBD data for {len(bg_nbd_data)} customers")

    # Prepare Gamma-Gamma inputs (only repeat customers)
    logger.info("Preparing Gamma-Gamma model inputs")
    gg_data = prepare_gamma_gamma_inputs(all_periods, min_frequency=args.min_frequency)

    logger.info(
        f"Prepared Gamma-Gamma data for {len(gg_data)} repeat customers "
        f"(min_frequency={args.min_frequency})"
    )

    # Convert Decimal columns to float for model compatibility
    for col in bg_nbd_data.select_dtypes(include=["object"]).columns:
        bg_nbd_data[col] = pd.to_numeric(bg_nbd_data[col], errors="ignore")

    if not gg_data.empty:
        for col in gg_data.select_dtypes(include=["object"]).columns:
            gg_data[col] = pd.to_numeric(gg_data[col], errors="ignore")

    # Fit BG/NBD model
    logger.info("Fitting BG/NBD model (purchase frequency)")
    bg_nbd_config = BGNBDConfig(method="map")
    bg_nbd_model = BGNBDModelWrapper(bg_nbd_config)
    bg_nbd_model.fit(bg_nbd_data)
    logger.info("BG/NBD model fitted successfully")

    # Fit Gamma-Gamma model
    logger.info("Fitting Gamma-Gamma model (monetary value)")
    gg_config = GammaGammaConfig(method="map")
    gg_model = GammaGammaModelWrapper(gg_config)

    if not gg_data.empty:
        gg_model.fit(gg_data)
        logger.info("Gamma-Gamma model fitted successfully")
    else:
        logger.warning(
            "No repeat customers found. CLV scores will be based on frequency only."
        )

    # Calculate CLV scores
    logger.info(f"Calculating CLV scores (time_horizon={args.time_horizon} months)")
    calculator = CLVCalculator(
        bg_nbd_model=bg_nbd_model,
        gamma_gamma_model=gg_model,
        time_horizon_months=args.time_horizon,
        profit_margin=Decimal(str(args.profit_margin)),
        discount_rate=Decimal(str(args.discount_rate)),
    )

    clv_scores = calculator.calculate_clv(bg_nbd_data, gg_data)

    # Export to CSV
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clv_scores.to_csv(output_path, index=False)
    logger.info(f"CLV scores exported to {output_path}")
    logger.info(
        f"Generated {len(clv_scores)} customer scores. "
        f"Top CLV: ${clv_scores['clv'].max():.2f}, "
        f"Average CLV: ${clv_scores['clv'].mean():.2f}"
    )

    return 0


def generate_five_lenses_report_cli(argv: list[str] | None = None) -> int:
    """Generate comprehensive Five Lenses customer base audit report.

    This command performs a complete customer base audit following the Five Lenses framework:
    - Lens 1: Single period analysis (current state metrics)
    - Lens 2: Period-to-period comparison (growth and migration patterns)
    - Lens 3: Cohort evolution (cohort performance over time)
    - Lens 4: Cohort comparison (relative cohort quality)
    - Lens 5: Overall customer base health (integrated health score)

    The report is exported as a Markdown file with all analysis results.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Generate Five Lenses customer base audit report"
    )
    parser.add_argument(
        "input", type=Path, help="Path to JSON file with raw transactions"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for output Markdown report",
    )
    parser.add_argument(
        "--period-granularity",
        type=str,
        choices=["month", "quarter", "year"],
        default="quarter",
        help="Period granularity for analysis (default: quarter)",
    )
    parser.add_argument(
        "--observation-end",
        type=str,
        help="Observation end date (ISO format: YYYY-MM-DD). Defaults to last transaction date.",
    )

    args = parser.parse_args(argv)

    # Load transactions
    logger.info(f"Loading transactions from {args.input}")
    transactions = _load_transactions(args.input)

    if not transactions:
        logger.error("No transactions found in input file")
        return 1

    # Build data mart
    granularity_map = {
        "month": PeriodGranularity.MONTH,
        "quarter": PeriodGranularity.QUARTER,
        "year": PeriodGranularity.YEAR,
    }
    granularity = granularity_map[args.period_granularity]

    logger.info(
        f"Building data mart from {len(transactions)} transactions "
        f"(granularity={granularity.value})"
    )
    builder = CustomerDataMartBuilder(period_granularities=[granularity])
    mart = builder.build(transactions)

    if granularity not in mart.periods or not mart.periods[granularity]:
        logger.error(f"No {granularity.value} periods generated from transactions")
        return 1

    periods = mart.periods[granularity]
    logger.info(f"Generated {len(periods)} {granularity.value} periods")

    # Determine observation end
    if args.observation_end:
        observation_end = datetime.fromisoformat(args.observation_end).replace(
            tzinfo=timezone.utc
        )
    else:
        observation_end = max(order.order_ts for order in mart.orders)

    logger.info(f"Observation end: {observation_end.date()}")

    # Extract unique periods (sorted by period_end)
    unique_period_ends = sorted(set(p.period_end for p in periods))

    if len(unique_period_ends) < 2:
        logger.error(
            "Need at least 2 periods for Five Lenses analysis. "
            f"Found {len(unique_period_ends)} period(s)."
        )
        return 1

    # ==============================================================================
    # Lens 1: Single Period Analysis (most recent period)
    # ==============================================================================
    logger.info("Running Lens 1: Single Period Analysis")

    recent_period_end = unique_period_ends[-1]
    recent_periods = [p for p in periods if p.period_end == recent_period_end]
    recent_rfm = calculate_rfm(
        recent_periods,
        observation_end=recent_period_end.replace(
            hour=23, minute=59, second=59, microsecond=999999
        ),
    )
    recent_scores = calculate_rfm_scores(recent_rfm)
    lens1_recent = analyze_single_period(recent_rfm, rfm_scores=recent_scores)

    logger.info(
        f"Lens 1: {lens1_recent.total_customers} customers, "
        f"${lens1_recent.total_revenue} revenue"
    )

    # ==============================================================================
    # Lens 2: Period-to-Period Comparison (previous vs recent)
    # ==============================================================================
    logger.info("Running Lens 2: Period-to-Period Comparison")

    prev_period_end = unique_period_ends[-2]
    prev_periods = [p for p in periods if p.period_end == prev_period_end]
    prev_rfm = calculate_rfm(
        prev_periods,
        observation_end=prev_period_end.replace(
            hour=23, minute=59, second=59, microsecond=999999
        ),
    )
    prev_scores = calculate_rfm_scores(prev_rfm)
    lens1_prev = analyze_single_period(prev_rfm, rfm_scores=prev_scores)

    lens2 = analyze_period_comparison(
        period1_rfm=prev_rfm,
        period2_rfm=recent_rfm,
        period1_metrics=lens1_prev,
        period2_metrics=lens1_recent,
    )

    logger.info(
        f"Lens 2: Retention={lens2.retention_rate}%, "
        f"Churn={lens2.churn_rate}%, "
        f"New customers={len(lens2.migration.new)}"
    )

    # ==============================================================================
    # Lens 3 & 4: Cohort Evolution and Comparison
    # ==============================================================================
    logger.info("Running Lens 3 & 4: Cohort Analysis")

    # Identify first transaction date for each customer
    customer_first_dates = {}
    for order in mart.orders:
        cust_id = order.customer_id
        if cust_id not in customer_first_dates:
            customer_first_dates[cust_id] = order.order_ts
        else:
            customer_first_dates[cust_id] = min(
                customer_first_dates[cust_id], order.order_ts
            )

    # Create cohort definitions based on period boundaries
    cohort_definitions = []
    for i, period_end in enumerate(unique_period_ends):
        # Find corresponding period_start
        period_records = [p for p in periods if p.period_end == period_end]
        if period_records:
            period_start = period_records[0].period_start
            cohort_id = f"cohort_{i + 1}_{period_start.strftime('%Y-%m')}"
            cohort_definitions.append(
                CohortDefinition(
                    cohort_id=cohort_id,
                    start_date=period_start,
                    end_date=period_end,
                    metadata={"period_number": i + 1},
                )
            )

    # Assign customers to cohorts
    customers = [
        CustomerIdentifier(cust_id, first_date, "transactions")
        for cust_id, first_date in customer_first_dates.items()
    ]

    cohort_assignments = assign_cohorts(
        customers, cohort_definitions, require_full_coverage=False
    )

    # Run Lens 3 for each cohort
    lens3_results = {}
    for cohort_def in cohort_definitions:
        cohort_customers = [
            cid
            for cid, cohort_id in cohort_assignments.items()
            if cohort_id == cohort_def.cohort_id
        ]

        if not cohort_customers:
            continue

        lens3 = analyze_cohort_evolution(
            cohort_name=cohort_def.cohort_id,
            acquisition_date=cohort_def.start_date,
            period_aggregations=periods,
            cohort_customer_ids=cohort_customers,
        )
        lens3_results[cohort_def.cohort_id] = lens3

    logger.info(f"Lens 3: Analyzed {len(lens3_results)} cohorts")

    # Note: Lens 4 (compare_cohorts) requires different input structure
    # For now, we skip detailed cohort comparison in the CLI report
    # Users can use the Python API for full Lens 4 analysis
    if len(lens3_results) >= 2:
        logger.info(f"Lens 4: {len(lens3_results)} cohorts available for comparison")
    else:
        logger.warning("Lens 4: Need at least 2 cohorts for comparison")

    # ==============================================================================
    # Lens 5: Overall Customer Base Health
    # ==============================================================================
    logger.info("Running Lens 5: Overall Customer Base Health")

    # Get earliest and latest periods for analysis dates
    analysis_start = min(p.period_start for p in periods)
    analysis_end = max(p.period_end for p in periods)

    lens5 = assess_customer_base_health(
        period_aggregations=periods,
        cohort_assignments=cohort_assignments,
        analysis_start_date=analysis_start,
        analysis_end_date=analysis_end,
    )

    logger.info(
        f"Lens 5: Health score={float(lens5.health_score.health_score):.2f}, "
        f"Grade={lens5.health_score.health_grade}"
    )

    # ==============================================================================
    # Generate Markdown Report
    # ==============================================================================
    logger.info("Generating Markdown report")

    report_lines = []
    report_lines.append("# Five Lenses Customer Base Audit Report\n")
    report_lines.append(
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    report_lines.append(
        f"**Observation Period:** {observation_end.strftime('%Y-%m-%d')}"
    )
    report_lines.append(f"**Analysis Granularity:** {granularity.value}\n")

    # Lens 1
    report_lines.append("## Lens 1: Single Period Analysis (Most Recent Period)\n")
    report_lines.append(f"- **Total Customers:** {lens1_recent.total_customers}")
    report_lines.append(f"- **Total Revenue:** ${lens1_recent.total_revenue}")
    report_lines.append(f"- **One-Time Buyers:** {lens1_recent.one_time_buyers}")
    report_lines.append(f"- **One-Time Buyer %:** {lens1_recent.one_time_buyer_pct}%")
    report_lines.append(
        f"- **Average Orders/Customer:** {lens1_recent.avg_orders_per_customer}\n"
    )

    # Lens 2
    report_lines.append("## Lens 2: Period-to-Period Comparison\n")
    report_lines.append(f"- **Retention Rate:** {lens2.retention_rate}%")
    report_lines.append(f"- **Churn Rate:** {lens2.churn_rate}%")
    report_lines.append(f"- **Retained Customers:** {len(lens2.migration.retained)}")
    report_lines.append(f"- **Churned Customers:** {len(lens2.migration.churned)}")
    report_lines.append(f"- **New Customers:** {len(lens2.migration.new)}")
    report_lines.append(f"- **Customer Count Change:** {lens2.customer_count_change}")
    report_lines.append(f"- **Revenue Change %:** {lens2.revenue_change_pct}%\n")

    # Lens 3
    report_lines.append("## Lens 3: Cohort Evolution\n")
    for cohort_id, lens3 in lens3_results.items():
        report_lines.append(f"### {cohort_id}")
        report_lines.append(f"- **Cohort Size:** {lens3.cohort_size}")
        report_lines.append(f"- **Periods Tracked:** {len(lens3.periods)}")
        if lens3.periods:
            latest_period = lens3.periods[-1]
            report_lines.append(
                f"- **Latest Period Active Customers:** {latest_period.active_customers}"
            )
        report_lines.append("")

    # Lens 4 (skipped in CLI for simplicity)
    report_lines.append("## Lens 4: Cohort Comparison\n")
    report_lines.append(f"- **Cohorts Available:** {len(lens3_results)}")
    report_lines.append(
        "- **Note:** Detailed cohort comparison available via Python API\n"
    )

    # Lens 5
    report_lines.append("## Lens 5: Overall Customer Base Health\n")
    report_lines.append(
        f"- **Health Score:** {float(lens5.health_score.health_score):.2f} / 100"
    )
    report_lines.append(f"- **Health Grade:** {lens5.health_score.health_grade}")
    report_lines.append(f"- **Total Customers:** {lens5.health_score.total_customers}")

    # Write report
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Five Lenses report exported to {output_path}")
    logger.info(
        f"Report covers {len(unique_period_ends)} periods, "
        f"{len(lens3_results)} cohorts, "
        f"Health Grade: {lens5.health_score.health_grade}"
    )

    return 0


def main() -> None:
    raise SystemExit(build_data_mart_cli())


if __name__ == "__main__":  # pragma: no cover
    main()
