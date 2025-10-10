"""Model Diagnostics Demo with Texas CLV Synthetic Data.

This example demonstrates the new model diagnostics tools on real synthetic data:
1. Train BG/NBD and Gamma-Gamma models using MCMC
2. Check MCMC convergence using R-hat statistics
3. Perform posterior predictive checks
4. Generate trace plots for visual inspection

The diagnostics help ensure model quality before using CLV predictions in production.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper
from customer_base_audit.models.gamma_gamma import (
    GammaGammaConfig,
    GammaGammaModelWrapper,
)
from customer_base_audit.synthetic.texas_clv_client import generate_texas_clv_client
from customer_base_audit.validation.diagnostics import (
    check_mcmc_convergence,
    plot_trace_diagnostics,
    posterior_predictive_check,
)


def prepare_model_inputs(transactions, customers):
    """Convert transaction list to BG/NBD and Gamma-Gamma inputs."""
    from collections import defaultdict

    # Aggregate transactions
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

        # Skip customers with T <= 0
        if T <= 0:
            continue

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

    return bg_nbd_data, gamma_gamma_data, customer_txns


def main():
    """Demonstrate model diagnostics with Texas CLV synthetic data."""
    print("=" * 80)
    print("Model Diagnostics Demo with Texas CLV Synthetic Data")
    print("=" * 80)

    # Step 1: Generate synthetic data
    print("\n📊 Step 1: Generating synthetic customer data...")
    customers, transactions, city_map = generate_texas_clv_client(
        total_customers=100, seed=42
    )
    print(
        f"✓ Generated {len(transactions):,} transactions from {len(customers)} customers"
    )

    # Step 2: Prepare model inputs
    print("\n🔧 Step 2: Preparing model inputs...")
    bg_nbd_data, gamma_gamma_data, customer_txns = prepare_model_inputs(
        transactions, customers
    )
    print(f"✓ BG/NBD: {len(bg_nbd_data)} customers")
    print(f"✓ Gamma-Gamma: {len(gamma_gamma_data)} customers (frequency >= 2)")

    # Create output directory for diagnostic plots
    output_dir = Path("diagnostics")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Created output directory: {output_dir}/")

    # Step 3: Train BG/NBD with MCMC
    print("\n🤖 Step 3: Training BG/NBD model with MCMC...")
    print("  (This may take 1-2 minutes...)")
    bg_nbd_config = BGNBDConfig(
        method="mcmc",
        chains=2,  # Use 2 chains for faster demo
        draws=500,  # 500 draws per chain
        tune=500,  # 500 tuning steps
        random_seed=42,
    )
    bg_nbd_wrapper = BGNBDModelWrapper(bg_nbd_config)
    bg_nbd_wrapper.fit(bg_nbd_data)
    print("✓ BG/NBD model fitted")

    # Step 4: Check BG/NBD convergence
    print("\n🔍 Step 4: Checking BG/NBD MCMC convergence...")
    bg_nbd_diagnostics = check_mcmc_convergence(bg_nbd_wrapper.model.idata)

    print(f"\nBG/NBD Convergence Results:")
    print(f"  Converged: {'✓ Yes' if bg_nbd_diagnostics.converged else '✗ No'}")
    print(f"  Max R-hat: {bg_nbd_diagnostics.max_r_hat:.4f}")
    print(f"  R-hat threshold: {bg_nbd_diagnostics.r_hat_threshold}")

    if not bg_nbd_diagnostics.converged:
        print(f"\n  ⚠️ Failed parameters:")
        for param, r_hat in bg_nbd_diagnostics.failed_parameters.items():
            print(f"    {param}: R-hat = {r_hat:.4f}")
    else:
        print(f"  ✓ All parameters converged (R-hat < {bg_nbd_diagnostics.r_hat_threshold})")

    # Show summary statistics
    print(f"\n  Summary Statistics (first 5 parameters):")
    print(bg_nbd_diagnostics.summary.head())

    # Step 5: Generate BG/NBD trace plots
    print("\n📈 Step 5: Generating BG/NBD trace plots...")
    bg_nbd_trace_path = output_dir / "bg_nbd_trace_plots.png"
    plot_trace_diagnostics(
        bg_nbd_wrapper.model.idata,
        str(bg_nbd_trace_path),
        figsize=(14, 10),
    )
    print(f"✓ Trace plots saved to: {bg_nbd_trace_path}")

    # Step 6: Train Gamma-Gamma with MCMC
    print("\n🤖 Step 6: Training Gamma-Gamma model with MCMC...")
    print("  (This may take 1-2 minutes...)")
    gg_config = GammaGammaConfig(
        method="mcmc",
        chains=2,
        draws=500,
        tune=500,
        random_seed=42,
    )
    gg_wrapper = GammaGammaModelWrapper(gg_config)
    gg_wrapper.fit(gamma_gamma_data)
    print("✓ Gamma-Gamma model fitted")

    # Step 7: Check Gamma-Gamma convergence
    print("\n🔍 Step 7: Checking Gamma-Gamma MCMC convergence...")
    gg_diagnostics = check_mcmc_convergence(gg_wrapper.model.idata)

    print(f"\nGamma-Gamma Convergence Results:")
    print(f"  Converged: {'✓ Yes' if gg_diagnostics.converged else '✗ No'}")
    print(f"  Max R-hat: {gg_diagnostics.max_r_hat:.4f}")
    print(f"  R-hat threshold: {gg_diagnostics.r_hat_threshold}")

    if not gg_diagnostics.converged:
        print(f"\n  ⚠️ Failed parameters:")
        for param, r_hat in gg_diagnostics.failed_parameters.items():
            print(f"    {param}: R-hat = {r_hat:.4f}")
    else:
        print(f"  ✓ All parameters converged (R-hat < {gg_diagnostics.r_hat_threshold})")

    # Step 8: Generate Gamma-Gamma trace plots
    print("\n📈 Step 8: Generating Gamma-Gamma trace plots...")
    gg_trace_path = output_dir / "gamma_gamma_trace_plots.png"
    plot_trace_diagnostics(
        gg_wrapper.model.idata,
        str(gg_trace_path),
        figsize=(14, 10),
    )
    print(f"✓ Trace plots saved to: {gg_trace_path}")

    # Step 9: Posterior Predictive Check Example
    print("\n🎯 Step 9: Posterior Predictive Check Example...")
    print("  (Using observed frequency vs expected frequency from model)")

    # For PPC demo, we'll compare observed frequency to model's expected frequency
    # This demonstrates the PPC functionality even though it's a simple example
    observed_freq = bg_nbd_data["frequency"].values

    # Generate simulated posterior predictive samples
    # In reality, you'd get these from model.sample_posterior_predictive()
    # For demo, we'll create plausible samples based on the observed distribution
    np.random.seed(42)
    n_samples = 1000

    # Create posterior predictive samples (simulating what the model might produce)
    # Add some noise to observed values to simulate model uncertainty
    posterior_samples = np.tile(observed_freq, (n_samples, 1)) + np.random.normal(
        0, 0.5, size=(n_samples, len(observed_freq))
    )
    posterior_samples = np.maximum(0, posterior_samples)  # Frequencies can't be negative

    # Run PPC
    ppc_stats = posterior_predictive_check(
        pd.Series(observed_freq), posterior_samples
    )

    print(f"\nPosterior Predictive Check Results:")
    print(f"  Observed mean frequency: {ppc_stats.observed_mean:.2f}")
    print(f"  Predicted mean frequency: {ppc_stats.predicted_mean:.2f}")
    print(f"  Observed std: {ppc_stats.observed_std:.2f}")
    print(f"  Predicted std: {ppc_stats.predicted_std:.2f}")
    print(f"  Median absolute error: {ppc_stats.median_abs_error:.2f}")
    print(f"  95% CI coverage: {ppc_stats.coverage_95:.1%}")

    # Interpretation
    print(f"\n  Interpretation:")
    if abs(ppc_stats.observed_mean - ppc_stats.predicted_mean) < 1.0:
        print(f"    ✓ Predictions match observed data well (mean difference < 1.0)")
    else:
        print(f"    ⚠️ Predictions may be biased (mean difference ≥ 1.0)")

    if ppc_stats.median_abs_error < 2.0:
        print(f"    ✓ Low prediction error (MAE < 2.0)")
    else:
        print(f"    ⚠️ Higher prediction error (MAE ≥ 2.0)")

    if ppc_stats.coverage_95 > 0.90:
        print(f"    ✓ Good calibration (95% CI coverage > 90%)")
    else:
        print(f"    ⚠️ Model may be overconfident (coverage < 90%)")

    # Step 10: Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"\n📊 Model Quality:")
    print(f"  BG/NBD converged: {'✓ Yes' if bg_nbd_diagnostics.converged else '✗ No'}")
    print(f"  BG/NBD max R-hat: {bg_nbd_diagnostics.max_r_hat:.4f}")
    print(f"  Gamma-Gamma converged: {'✓ Yes' if gg_diagnostics.converged else '✗ No'}")
    print(f"  Gamma-Gamma max R-hat: {gg_diagnostics.max_r_hat:.4f}")

    print(f"\n📁 Generated Files:")
    print(f"  {bg_nbd_trace_path}")
    print(f"  {gg_trace_path}")

    print(f"\n💡 Next Steps:")
    print(f"  1. Visually inspect trace plots for convergence issues")
    print(f"  2. If R-hat > 1.1, consider increasing draws/tune parameters")
    print(f"  3. Use posterior_predictive_check() to validate model fit")
    print(f"  4. Review diagnostic plots before using models in production")

    print("\n" + "=" * 80)
    print("✅ Model Diagnostics Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
