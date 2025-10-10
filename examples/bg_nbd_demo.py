"""Test BG/NBD model with Texas CLV synthetic data."""

from datetime import datetime

from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper
from customer_base_audit.models.gamma_gamma import (
    GammaGammaConfig,
    GammaGammaModelWrapper,
)
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client


def main():
    """Demonstrate BG/NBD and Gamma-Gamma models on synthetic Texas CLV data."""
    print("=" * 80)
    print("BG/NBD + Gamma-Gamma Model Test with Texas CLV Synthetic Data")
    print("=" * 80)

    # Step 1: Generate synthetic data
    print("\n📊 Step 1: Generating synthetic customer data...")
    customers, transactions, city_map = generate_texas_clv_client(
        total_customers=200, seed=42
    )
    print(
        f"✓ Generated {len(transactions):,} transactions from {len(customers)} customers"
    )
    print(f"  Cities: {', '.join(sorted(set(city_map.values())))}")

    # Step 2: Convert transactions to simpler format for analysis
    print("\n📈 Step 2: Aggregating transaction data...")
    from collections import defaultdict

    # Aggregate transactions by customer
    customer_txns = defaultdict(
        lambda: {"orders": set(), "spend": 0.0, "first_date": None, "last_date": None}
    )
    for txn in transactions:
        cust_data = customer_txns[txn.customer_id]
        cust_data["orders"].add(txn.order_id)
        cust_data["spend"] += txn.quantity * txn.unit_price
        txn_date = (
            txn.event_ts.date() if hasattr(txn.event_ts, "date") else txn.event_ts
        )
        if cust_data["first_date"] is None or txn_date < cust_data["first_date"]:
            cust_data["first_date"] = txn_date
        if cust_data["last_date"] is None or txn_date > cust_data["last_date"]:
            cust_data["last_date"] = txn_date

    print(f"✓ Aggregated {len(customer_txns)} unique customers")
    print(f"  Total revenue: ${sum(c['spend'] for c in customer_txns.values()):,.2f}")

    # Step 3: Prepare model inputs manually
    print("\n🔧 Step 3: Preparing model inputs...")
    import pandas as pd

    observation_end = datetime(2024, 12, 31)

    # Create BG/NBD inputs manually
    bg_nbd_rows = []
    for cust_id, cust_data in customer_txns.items():
        num_orders = len(cust_data["orders"])
        frequency = max(0, num_orders - 1)  # Repeat purchases only

        # Calculate recency and T
        first_dt = datetime.combine(cust_data["first_date"], datetime.min.time())
        last_dt = datetime.combine(cust_data["last_date"], datetime.min.time())

        if frequency > 0:
            recency = (last_dt - first_dt).total_seconds() / 86400.0
        else:
            recency = 0.0

        T = (observation_end - first_dt).total_seconds() / 86400.0

        bg_nbd_rows.append(
            {"customer_id": cust_id, "frequency": frequency, "recency": recency, "T": T}
        )

    bg_nbd_data = pd.DataFrame(bg_nbd_rows)
    print(f"✓ BG/NBD inputs prepared for {len(bg_nbd_data)} customers")
    print(
        f"  Frequency range: {int(bg_nbd_data['frequency'].min())}-{int(bg_nbd_data['frequency'].max())}"
    )
    print(f"  Mean frequency: {bg_nbd_data['frequency'].mean():.2f}")

    # Create Gamma-Gamma inputs manually
    gg_rows = []
    for cust_id, cust_data in customer_txns.items():
        num_orders = len(cust_data["orders"])
        if num_orders >= 2:  # min_frequency
            monetary_value = cust_data["spend"] / num_orders
            gg_rows.append(
                {
                    "customer_id": cust_id,
                    "frequency": num_orders,
                    "monetary_value": monetary_value,
                }
            )

    gamma_gamma_data = pd.DataFrame(gg_rows)
    print(
        f"✓ Gamma-Gamma inputs prepared for {len(gamma_gamma_data)} customers (freq >= 2)"
    )
    print(
        f"  Monetary value range: ${gamma_gamma_data['monetary_value'].min():.2f}-${gamma_gamma_data['monetary_value'].max():.2f}"
    )
    print(f"  Mean monetary value: ${gamma_gamma_data['monetary_value'].mean():.2f}")

    # Step 4: Train BG/NBD model
    print("\n🤖 Step 4: Training BG/NBD model (purchase frequency)...")
    bg_nbd_config = BGNBDConfig(method="map")
    bg_nbd_wrapper = BGNBDModelWrapper(bg_nbd_config)
    bg_nbd_wrapper.fit(bg_nbd_data)
    print("✓ BG/NBD model trained successfully (MAP method)")

    # Step 5: Train Gamma-Gamma model
    print("\n💰 Step 5: Training Gamma-Gamma model (monetary value)...")
    gg_config = GammaGammaConfig(method="map")
    gg_wrapper = GammaGammaModelWrapper(gg_config)
    gg_wrapper.fit(gamma_gamma_data)
    print("✓ Gamma-Gamma model trained successfully (MAP method)")

    # Step 6: Generate predictions
    print("\n📊 Step 6: Generating predictions...")

    # BG/NBD predictions (6-month horizon)
    time_horizon_days = 180.0
    purchase_predictions = bg_nbd_wrapper.predict_purchases(
        bg_nbd_data, time_periods=time_horizon_days
    )
    prob_alive = bg_nbd_wrapper.calculate_probability_alive(bg_nbd_data)

    print(f"✓ Purchase predictions (next {int(time_horizon_days)} days):")
    print(
        f"  Range: {purchase_predictions['predicted_purchases'].min():.2f}-{purchase_predictions['predicted_purchases'].max():.2f}"
    )
    print(f"  Mean: {purchase_predictions['predicted_purchases'].mean():.2f}")

    print("✓ Probability alive:")
    print(
        f"  Range: {prob_alive['prob_alive'].min():.3f}-{prob_alive['prob_alive'].max():.3f}"
    )
    print(f"  Mean: {prob_alive['prob_alive'].mean():.3f}")

    # Gamma-Gamma predictions
    monetary_predictions = gg_wrapper.predict_spend(gamma_gamma_data)
    print("✓ Monetary value predictions:")
    print(
        f"  Range: ${monetary_predictions['predicted_monetary_value'].min():.2f}-${monetary_predictions['predicted_monetary_value'].max():.2f}"
    )
    print(f"  Mean: ${monetary_predictions['predicted_monetary_value'].mean():.2f}")

    # Step 7: Analyze by customer segments
    print("\n📈 Step 7: Customer segment analysis...")

    # Merge predictions with input data
    analysis = bg_nbd_data.merge(purchase_predictions, on="customer_id").merge(
        prob_alive, on="customer_id"
    )

    # Add monetary predictions for repeat customers
    analysis = analysis.merge(monetary_predictions, on="customer_id", how="left")

    # Calculate simple CLV for repeat customers
    # CLV = (predicted purchases) * (predicted avg value)
    analysis["estimated_clv_6mo"] = analysis["predicted_purchases"] * analysis[
        "predicted_monetary_value"
    ].fillna(0)

    # Segment analysis
    print("\n🎯 High-Frequency Customers (frequency >= 5):")
    high_freq = analysis[analysis["frequency"] >= 5]
    if len(high_freq) > 0:
        print(f"  Count: {len(high_freq)}")
        print(f"  Mean P(alive): {high_freq['prob_alive'].mean():.3f}")
        print(
            f"  Mean predicted purchases: {high_freq['predicted_purchases'].mean():.2f}"
        )
        clv_with_monetary = high_freq[high_freq["estimated_clv_6mo"] > 0]
        if len(clv_with_monetary) > 0:
            print(
                f"  Mean 6-month CLV: ${clv_with_monetary['estimated_clv_6mo'].mean():.2f}"
            )

    print("\n🎯 One-Time Buyers (frequency = 0):")
    one_time = analysis[analysis["frequency"] == 0]
    if len(one_time) > 0:
        print(f"  Count: {len(one_time)}")
        print(f"  Mean P(alive): {one_time['prob_alive'].mean():.3f}")
        print(
            f"  Mean predicted purchases: {one_time['predicted_purchases'].mean():.2f}"
        )

    print("\n🎯 Medium-Frequency Customers (2 <= frequency < 5):")
    medium_freq = analysis[(analysis["frequency"] >= 2) & (analysis["frequency"] < 5)]
    if len(medium_freq) > 0:
        print(f"  Count: {len(medium_freq)}")
        print(f"  Mean P(alive): {medium_freq['prob_alive'].mean():.3f}")
        print(
            f"  Mean predicted purchases: {medium_freq['predicted_purchases'].mean():.2f}"
        )
        clv_with_monetary = medium_freq[medium_freq["estimated_clv_6mo"] > 0]
        if len(clv_with_monetary) > 0:
            print(
                f"  Mean 6-month CLV: ${clv_with_monetary['estimated_clv_6mo'].mean():.2f}"
            )

    # Step 8: Top 10 customers by CLV
    print("\n🏆 Top 10 Customers by Estimated 6-Month CLV:")
    top_customers = analysis[analysis["estimated_clv_6mo"] > 0].nlargest(
        10, "estimated_clv_6mo"
    )[
        [
            "customer_id",
            "frequency",
            "prob_alive",
            "predicted_purchases",
            "predicted_monetary_value",
            "estimated_clv_6mo",
        ]
    ]
    for idx, row in top_customers.iterrows():
        print(
            f"  {row['customer_id']}: "
            f"freq={int(row['frequency'])}, "
            f"P(alive)={row['prob_alive']:.3f}, "
            f"pred_purch={row['predicted_purchases']:.1f}, "
            f"pred_value=${row['predicted_monetary_value']:.2f}, "
            f"CLV=${row['estimated_clv_6mo']:.2f}"
        )

    print("\n" + "=" * 80)
    print("✅ BG/NBD + Gamma-Gamma Model Test Complete!")
    print("=" * 80)
    print("\n📋 Summary:")
    print(f"  • Total customers analyzed: {len(analysis)}")
    print(
        f"  • Customers with CLV estimates: {len(analysis[analysis['estimated_clv_6mo'] > 0])}"
    )
    print(
        f"  • Total estimated 6-month revenue: ${analysis['estimated_clv_6mo'].sum():.2f}"
    )
    print(f"  • Average P(alive): {analysis['prob_alive'].mean():.3f}")
    print(
        f"  • Average predicted purchases (6mo): {analysis['predicted_purchases'].mean():.2f}"
    )


if __name__ == "__main__":
    main()
