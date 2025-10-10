"""Complete CLV calculation demo with Texas CLV synthetic data.

This example demonstrates the full CLV calculation pipeline:
1. Generate synthetic customer transactions
2. Prepare BG/NBD and Gamma-Gamma model inputs
3. Train both models
4. Calculate CLV scores
5. Analyze customer segments by value
"""

from datetime import datetime
from decimal import Decimal

from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper
from customer_base_audit.models.clv_calculator import CLVCalculator
from customer_base_audit.models.gamma_gamma import (
    GammaGammaConfig,
    GammaGammaModelWrapper,
)
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client


def main():
    """Demonstrate complete CLV calculation pipeline."""
    print("=" * 80)
    print("Complete CLV Calculation Demo with Texas CLV Synthetic Data")
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

    # Step 2: Aggregate transaction data
    print("\n📈 Step 2: Aggregating transaction data...")
    from collections import defaultdict

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

    # Step 3: Prepare model inputs
    print("\n🔧 Step 3: Preparing model inputs...")
    import pandas as pd

    observation_end = datetime(2024, 12, 31)

    # BG/NBD inputs
    bg_nbd_rows = []
    for cust_id, cust_data in customer_txns.items():
        num_orders = len(cust_data["orders"])
        frequency = max(0, num_orders - 1)

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

    # Gamma-Gamma inputs
    gg_rows = []
    for cust_id, cust_data in customer_txns.items():
        num_orders = len(cust_data["orders"])
        if num_orders >= 2:
            monetary_value = cust_data["spend"] / num_orders
            gg_rows.append(
                {
                    "customer_id": cust_id,
                    "frequency": num_orders,
                    "monetary_value": monetary_value,
                }
            )

    gamma_gamma_data = pd.DataFrame(gg_rows)

    print(f"✓ BG/NBD inputs prepared for {len(bg_nbd_data)} customers")
    print(
        f"  Frequency range: {int(bg_nbd_data['frequency'].min())}-{int(bg_nbd_data['frequency'].max())}"
    )
    print(
        f"✓ Gamma-Gamma inputs prepared for {len(gamma_gamma_data)} customers (freq >= 2)"
    )

    # Step 4: Train models
    print("\n🤖 Step 4: Training CLV models...")
    bg_nbd_config = BGNBDConfig(method="map")
    bg_nbd_wrapper = BGNBDModelWrapper(bg_nbd_config)
    bg_nbd_wrapper.fit(bg_nbd_data)
    print("✓ BG/NBD model trained (purchase frequency)")

    gg_config = GammaGammaConfig(method="map")
    gg_wrapper = GammaGammaModelWrapper(gg_config)
    gg_wrapper.fit(gamma_gamma_data)
    print("✓ Gamma-Gamma model trained (monetary value)")

    # Step 5: Calculate CLV
    print("\n💰 Step 5: Calculating CLV scores...")
    calculator = CLVCalculator(
        bg_nbd_model=bg_nbd_wrapper,
        gamma_gamma_model=gg_wrapper,
        time_horizon_months=12,  # 1-year CLV
        discount_rate=Decimal("0.10"),  # 10% annual discount rate
        profit_margin=Decimal("0.30"),  # 30% profit margin
    )

    clv_scores = calculator.calculate_clv(bg_nbd_data, gamma_gamma_data)
    print(f"✓ CLV calculated for {len(clv_scores)} customers")

    # Step 6: Analyze results
    print("\n📊 Step 6: CLV Analysis...")
    print(f"\n💵 Overall Metrics:")
    print(f"  Total 1-year CLV: ${clv_scores['clv'].sum():,.2f}")
    print(f"  Average CLV: ${clv_scores['clv'].mean():.2f}")
    print(f"  Median CLV: ${clv_scores['clv'].median():.2f}")
    print(f"  Max CLV: ${clv_scores['clv'].max():.2f}")
    print(f"  Min CLV: ${clv_scores['clv'].min():.2f}")

    # Average metrics
    print(f"\n📈 Average Predictions:")
    print(
        f"  Avg predicted purchases (12mo): {clv_scores['predicted_purchases'].mean():.2f}"
    )
    print(f"  Avg transaction value: ${clv_scores['predicted_avg_value'].mean():.2f}")
    print(f"  Avg probability alive: {clv_scores['prob_alive'].mean():.3f}")

    # Top 10 customers
    print("\n🏆 Top 10 Customers by CLV:")
    top_10 = clv_scores.head(10)
    for idx, row in top_10.iterrows():
        print(
            f"  {idx + 1}. {row['customer_id']}: "
            f"CLV=${row['clv']:.2f}, "
            f"P(alive)={row['prob_alive']:.3f}, "
            f"pred_purch={row['predicted_purchases']:.1f}, "
            f"avg_value=${row['predicted_avg_value']:.2f}"
        )

    # Customer segments
    print("\n🎯 Customer Segmentation by CLV:")

    # High-value customers (top 10%)
    high_value_threshold = clv_scores["clv"].quantile(0.90)
    high_value = clv_scores[clv_scores["clv"] >= high_value_threshold]
    print(f"\n  High-Value (Top 10%, CLV >= ${high_value_threshold:.2f}):")
    print(f"    Count: {len(high_value)} customers")
    print(f"    Total CLV: ${high_value['clv'].sum():.2f}")
    print(
        f"    % of total revenue: {(high_value['clv'].sum() / clv_scores['clv'].sum() * 100):.1f}%"
    )
    print(f"    Avg P(alive): {high_value['prob_alive'].mean():.3f}")

    # Medium-value customers (40-90th percentile)
    medium_value_low = clv_scores["clv"].quantile(0.40)
    medium_value = clv_scores[
        (clv_scores["clv"] >= medium_value_low)
        & (clv_scores["clv"] < high_value_threshold)
    ]
    print(
        f"\n  Medium-Value (40-90th percentile, ${medium_value_low:.2f}-${high_value_threshold:.2f}):"
    )
    print(f"    Count: {len(medium_value)} customers")
    print(f"    Total CLV: ${medium_value['clv'].sum():.2f}")
    print(
        f"    % of total revenue: {(medium_value['clv'].sum() / clv_scores['clv'].sum() * 100):.1f}%"
    )
    print(f"    Avg P(alive): {medium_value['prob_alive'].mean():.3f}")

    # Low-value customers (bottom 40%)
    low_value = clv_scores[clv_scores["clv"] < medium_value_low]
    print(f"\n  Low-Value (Bottom 40%, CLV < ${medium_value_low:.2f}):")
    print(f"    Count: {len(low_value)} customers")
    print(f"    Total CLV: ${low_value['clv'].sum():.2f}")
    print(
        f"    % of total revenue: {(low_value['clv'].sum() / clv_scores['clv'].sum() * 100):.1f}%"
    )
    print(f"    Avg P(alive): {low_value['prob_alive'].mean():.3f}")

    # Zero CLV customers (one-time buyers)
    zero_clv = clv_scores[clv_scores["clv"] == 0]
    print(f"\n  Zero CLV (One-time buyers):")
    print(f"    Count: {len(zero_clv)} customers")
    print(f"    % of customer base: {(len(zero_clv) / len(clv_scores) * 100):.1f}%")

    # Step 7: Business insights
    print("\n💡 Step 7: Business Insights...")
    print(f"\n  📊 Revenue Concentration:")
    top_20_pct = clv_scores.head(int(len(clv_scores) * 0.2))
    print(
        f"    Top 20% of customers generate {(top_20_pct['clv'].sum() / clv_scores['clv'].sum() * 100):.1f}% of CLV"
    )

    print(f"\n  🎯 Recommended Actions:")
    if len(high_value) > 0:
        print(
            f"    • Prioritize retention for {len(high_value)} high-value customers (${high_value['clv'].sum():.2f} at risk)"
        )
    if len(zero_clv) > 0:
        print(
            f"    • Reactivation campaign for {len(zero_clv)} one-time buyers ({(len(zero_clv) / len(clv_scores) * 100):.1f}% of base)"
        )
    avg_prob_alive = clv_scores["prob_alive"].mean()
    if avg_prob_alive < 0.7:
        print(
            f"    • Address churn risk (avg P(alive) = {avg_prob_alive:.3f}, target: >0.70)"
        )

    print("\n" + "=" * 80)
    print("✅ CLV Calculation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
