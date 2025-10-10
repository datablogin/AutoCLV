"""Validation Framework Demo with Texas CLV Synthetic Data.

This example demonstrates the new validation framework on real synthetic data:
1. Generate Texas CLV synthetic customer data
2. Perform temporal train/test splitting
3. Train BG/NBD and Gamma-Gamma models
4. Calculate validation metrics (MAE, MAPE, RMSE, ARPE, R²)
5. Run time-series cross-validation with expanding windows

The validation framework helps assess CLV model performance before production deployment.
"""

from datetime import datetime

import pandas as pd

from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper
from customer_base_audit.models.gamma_gamma import (
    GammaGammaConfig,
    GammaGammaModelWrapper,
)
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client
from customer_base_audit.validation.validation import (
    calculate_clv_metrics,
    temporal_train_test_split,
)


def prepare_rfm_data(transactions, observation_end_date):
    """Convert transaction DataFrame to RFM DataFrame.

    Parameters
    ----------
    transactions: pd.DataFrame
        DataFrame with columns: customer_id, event_ts, amount
    observation_end_date: datetime
        End of observation period

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (bg_nbd_data, gamma_gamma_data)
    """
    # Vectorized aggregation instead of iterrows() for better performance
    # Convert event_ts to date if needed
    transactions = transactions.copy()
    if hasattr(transactions["event_ts"].iloc[0], "date"):
        transactions["event_date"] = transactions["event_ts"].dt.date
    else:
        transactions["event_date"] = transactions["event_ts"]

    # Group by customer and aggregate statistics
    customer_stats = transactions.groupby("customer_id").agg(
        txn_count=("customer_id", "size"),
        spend=("amount", "sum"),
        first_date=("event_date", "min"),
        last_date=("event_date", "max"),
    )

    # BG/NBD inputs - vectorized calculations
    bg_nbd_data = customer_stats.reset_index()
    bg_nbd_data["frequency"] = (bg_nbd_data["txn_count"] - 1).clip(lower=0)

    # Calculate recency in days
    bg_nbd_data["first_datetime"] = pd.to_datetime(
        bg_nbd_data["first_date"].apply(
            lambda d: datetime.combine(d, datetime.min.time())
        )
    )
    bg_nbd_data["last_datetime"] = pd.to_datetime(
        bg_nbd_data["last_date"].apply(
            lambda d: datetime.combine(d, datetime.min.time())
        )
    )

    bg_nbd_data["recency"] = (
        bg_nbd_data["last_datetime"] - bg_nbd_data["first_datetime"]
    ).dt.total_seconds() / 86400.0
    bg_nbd_data.loc[bg_nbd_data["frequency"] == 0, "recency"] = 0.0

    # Calculate T (age) in days
    bg_nbd_data["T"] = (
        pd.to_datetime(observation_end_date) - bg_nbd_data["first_datetime"]
    ).dt.total_seconds() / 86400.0

    # Filter out customers with T <= 0
    bg_nbd_data = bg_nbd_data[bg_nbd_data["T"] > 0]

    # Select final columns
    bg_nbd_data = bg_nbd_data[["customer_id", "frequency", "recency", "T"]]

    # Gamma-Gamma inputs - only customers with 2+ transactions
    gamma_gamma_data = customer_stats[customer_stats["txn_count"] >= 2].copy()
    gamma_gamma_data["monetary_value"] = (
        gamma_gamma_data["spend"] / gamma_gamma_data["txn_count"]
    )
    gamma_gamma_data = gamma_gamma_data.reset_index()[
        ["customer_id", "txn_count", "monetary_value"]
    ]
    gamma_gamma_data = gamma_gamma_data.rename(columns={"txn_count": "frequency"})

    return bg_nbd_data, gamma_gamma_data


def transactions_to_dataframe(transactions):
    """Convert transaction list to DataFrame."""
    return pd.DataFrame(
        [
            {
                "customer_id": txn.customer_id,
                "event_ts": pd.to_datetime(txn.event_ts),
                "amount": txn.quantity * txn.unit_price,
            }
            for txn in transactions
        ]
    )


def clv_pipeline(train_txns, train_end_date):
    """Simple CLV prediction pipeline using BG/NBD + Gamma-Gamma.

    Parameters
    ----------
    train_txns: pd.DataFrame
        Training transactions with columns: customer_id, event_ts, amount
    train_end_date: datetime
        End of training period

    Returns
    -------
    pd.DataFrame
        Predictions with columns: customer_id, clv
    """
    # Prepare RFM data from training transactions
    bg_nbd_data, gamma_gamma_data = prepare_rfm_data(train_txns, train_end_date)

    # Train BG/NBD model (using MAP for speed)
    bg_nbd_config = BGNBDConfig(method="map")
    bg_nbd_wrapper = BGNBDModelWrapper(bg_nbd_config)
    bg_nbd_wrapper.fit(bg_nbd_data)

    # Train Gamma-Gamma model
    gg_config = GammaGammaConfig(method="map")
    gg_wrapper = GammaGammaModelWrapper(gg_config)
    gg_wrapper.fit(gamma_gamma_data)

    # Predict CLV for next 90 days
    # Get purchase predictions
    purchase_pred = bg_nbd_wrapper.predict_purchases(bg_nbd_data, time_periods=90.0)

    # Get spend predictions
    spend_pred = gg_wrapper.predict_spend(gamma_gamma_data)

    # Merge predictions
    predictions = purchase_pred.merge(
        spend_pred[["customer_id", "predicted_monetary_value"]],
        on="customer_id",
        how="left",
    )

    # Fill missing spend predictions with historical average
    train_df = (
        pd.DataFrame(train_txns)
        if not isinstance(train_txns, pd.DataFrame)
        else train_txns
    )
    for idx, row in predictions.iterrows():
        if pd.isna(row["predicted_monetary_value"]):
            cust_id = row["customer_id"]
            cust_txns = train_df[train_df["customer_id"] == cust_id]
            predictions.at[idx, "predicted_monetary_value"] = (
                cust_txns["amount"].mean() if len(cust_txns) > 0 else 0.0
            )

    # Calculate CLV
    predictions["clv"] = (
        predictions["predicted_purchases"] * predictions["predicted_monetary_value"]
    )

    return predictions[["customer_id", "clv"]]


def main():
    """Demonstrate validation framework with Texas CLV synthetic data."""
    print("=" * 80)
    print("Validation Framework Demo with Texas CLV Synthetic Data")
    print("=" * 80)

    # Step 1: Generate synthetic data
    print("\n📊 Step 1: Generating synthetic customer data...")
    customers, transactions, city_map = generate_texas_clv_client(
        total_customers=200, seed=42
    )
    print(
        f"✓ Generated {len(transactions):,} transactions from {len(customers)} customers"
    )

    # Convert to DataFrame
    txn_df = transactions_to_dataframe(transactions)
    print(
        f"✓ Transaction date range: {txn_df['event_ts'].min()} to {txn_df['event_ts'].max()}"
    )

    # Step 2: Temporal Train/Test Split
    print("\n🔪 Step 2: Performing temporal train/test split...")
    train_end = datetime(2024, 9, 1)
    obs_end = datetime(2024, 12, 31)

    train_txns, obs_txns, test_txns = temporal_train_test_split(
        txn_df,
        train_end_date=train_end,
        observation_end_date=obs_end,
    )

    print(f"✓ Training transactions (before {train_end.date()}): {len(train_txns):,}")
    print(f"✓ Observation transactions (through {obs_end.date()}): {len(obs_txns):,}")
    print(
        f"✓ Test transactions ({train_end.date()} to {obs_end.date()}): {len(test_txns):,}"
    )

    # Step 3: Train models and make predictions
    print("\n🤖 Step 3: Training CLV models on training data...")
    print("  (Using MAP estimation for speed...)")

    # Prepare training data
    bg_nbd_data, gamma_gamma_data = prepare_rfm_data(train_txns, train_end)
    print(f"✓ Prepared RFM data: {len(bg_nbd_data)} customers for BG/NBD")
    print(f"✓ Prepared RFM data: {len(gamma_gamma_data)} customers for Gamma-Gamma")

    # Train BG/NBD
    bg_nbd_config = BGNBDConfig(method="map")
    bg_nbd_wrapper = BGNBDModelWrapper(bg_nbd_config)
    bg_nbd_wrapper.fit(bg_nbd_data)
    print("✓ BG/NBD model trained")

    # Train Gamma-Gamma
    gg_config = GammaGammaConfig(method="map")
    gg_wrapper = GammaGammaModelWrapper(gg_config)
    gg_wrapper.fit(gamma_gamma_data)
    print("✓ Gamma-Gamma model trained")

    # Step 4: Make predictions and calculate metrics
    print("\n📈 Step 4: Making predictions and calculating validation metrics...")

    # Calculate actual CLV from test period
    actual_clv = test_txns.groupby("customer_id")["amount"].sum()
    print(f"✓ Calculated actual CLV for {len(actual_clv)} customers")

    # Make predictions for test period (90 days)
    # Predict expected purchases for all customers
    purchase_pred = bg_nbd_wrapper.predict_purchases(bg_nbd_data, time_periods=90.0)

    # Predict average transaction values
    spend_pred = gg_wrapper.predict_spend(gamma_gamma_data)

    # Merge predictions
    predictions = purchase_pred.merge(
        spend_pred[["customer_id", "predicted_monetary_value"]],
        on="customer_id",
        how="left",
    )

    # Calculate CLV = expected purchases * average spend
    # For customers without Gamma-Gamma predictions, use historical average
    for idx, row in predictions.iterrows():
        if pd.isna(row["predicted_monetary_value"]):
            cust_id = row["customer_id"]
            cust_txns = train_txns[train_txns["customer_id"] == cust_id]
            predictions.at[idx, "predicted_monetary_value"] = (
                cust_txns["amount"].mean() if len(cust_txns) > 0 else 0.0
            )

    predictions["clv"] = (
        predictions["predicted_purchases"] * predictions["predicted_monetary_value"]
    )

    pred_df = pd.DataFrame(predictions).set_index("customer_id")
    print(f"✓ Generated predictions for {len(pred_df)} customers")

    # Align predictions with actual CLV
    comparison = pred_df.join(actual_clv.rename("actual_clv"), how="inner")
    print(f"✓ Aligned {len(comparison)} customers with both predictions and actuals")

    # Calculate validation metrics
    metrics = calculate_clv_metrics(
        actual=comparison["actual_clv"],
        predicted=comparison["clv"],
    )

    print("\n📊 Validation Metrics:")
    print(f"  MAE (Mean Absolute Error):        ${metrics.mae:>10}")
    print(f"  MAPE (Mean Abs Percentage Error): {metrics.mape:>10}%")
    print(f"  RMSE (Root Mean Squared Error):   ${metrics.rmse:>10}")
    print(f"  ARPE (Aggregate Revenue % Error): {metrics.arpe:>10}%")
    print(f"  R² (Coefficient of Determination): {metrics.r_squared:>10}")
    print(f"  Sample Size:                       {metrics.sample_size:>10}")

    # Interpretation
    print("\n  Target Performance:")
    mape_ok = metrics.mape < 20
    arpe_ok = metrics.arpe < 10
    r2_ok = metrics.r_squared > 0.5

    print(f"  MAPE < 20%: {'✓' if mape_ok else '✗'} ({metrics.mape}%)")
    print(f"  ARPE < 10%: {'✓' if arpe_ok else '✗'} ({metrics.arpe}%)")
    print(f"  R² > 0.5:   {'✓' if r2_ok else '✗'} ({metrics.r_squared})")

    if mape_ok and arpe_ok and r2_ok:
        print("\n  ✓ Model meets all target performance criteria!")
    else:
        print("\n  ⚠️ Model needs improvement on some metrics")

    # Step 5: Cross-Validation (Skipped in demo for speed)
    print("\n🔄 Step 5: Cross-Validation Framework")
    print("  Note: Cross-validation skipped in demo (would take 5-10 minutes)")
    print("  The cross_validate_clv() function supports:")
    print("  • Time-series cross-validation with expanding windows")
    print("  • Configurable n_folds, time increments, and initial training period")
    print("  • Custom model pipelines")
    print("\n  Example usage:")
    print("    cv_metrics = cross_validate_clv(")
    print("        transactions,")
    print("        clv_pipeline,")
    print("        n_folds=3,")
    print("        time_increment_months=3,")
    print("        initial_train_months=12")
    print("    )")
    print(
        "\n  This would train the model 3 times and return a list of ValidationMetrics"
    )

    # Step 6: Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\n📊 Single Hold-Out Validation:")
    print(f"  MAPE: {metrics.mape}% (target: < 20%)")
    print(f"  ARPE: {metrics.arpe}% (target: < 10%)")
    print(f"  R²:   {metrics.r_squared} (target: > 0.5)")

    print("\n💡 Key Insights:")
    print("  • Temporal train/test split respects time-series ordering")
    print("  • Validation metrics quantify model performance")
    print("  • Cross-validation provides robust performance estimates")
    print(
        "  • ARPE measures aggregate-level accuracy (important for revenue forecasting)"
    )
    print("  • MAPE measures individual-level accuracy (important for targeting)")

    print("\n💡 Next Steps:")
    print("  1. If metrics don't meet targets, tune model hyperparameters")
    print("  2. Try MCMC estimation for better uncertainty quantification")
    print("  3. Increase training data size for better model fit")
    print("  4. Consider feature engineering (cohort analysis, seasonality)")

    print("\n" + "=" * 80)
    print("✅ Validation Framework Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
