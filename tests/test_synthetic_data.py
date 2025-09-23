from datetime import date

from customer_base_audit.synthetic import (
    ScenarioConfig,
    check_non_negative_amounts,
    check_promo_spike_signal,
    check_reasonable_order_density,
    generate_customers,
    generate_transactions,
)


def test_generate_customers_and_transactions_basic() -> None:
    customers = generate_customers(50, date(2024, 1, 1), date(2024, 12, 31), seed=7)
    assert len(customers) == 50
    txns = generate_transactions(customers, date(2024, 1, 1), date(2024, 12, 31))
    assert isinstance(txns, list)
    # Not asserting an exact count; ensure plausible volume
    assert len(txns) > 0
    assert check_non_negative_amounts(txns).ok
    assert check_reasonable_order_density(txns, min_avg_lines_per_order=1.0).ok


def test_promo_spike_signal_detected() -> None:
    customers = generate_customers(40, date(2024, 1, 1), date(2024, 6, 30), seed=123)
    scenario = ScenarioConfig(promo_month=5, promo_uplift=2.0, seed=999)
    txns = generate_transactions(
        customers, date(2024, 1, 1), date(2024, 6, 30), scenario=scenario
    )

    result = check_promo_spike_signal(txns, promo_month=5, min_ratio=1.1)
    assert result.ok, result.message


def test_empty_inputs_are_handled() -> None:
    assert generate_customers(0, date(2024, 1, 1), date(2024, 1, 1)) == []
    assert check_non_negative_amounts([]).ok
    assert not check_promo_spike_signal([], promo_month=1).ok


def test_scenario_config_validation_and_edge_cases() -> None:
    # Invalid promo month
    from customer_base_audit.synthetic import ScenarioConfig

    try:
        ScenarioConfig(promo_month=0)
        assert False, "Expected ValueError for promo_month=0"
    except ValueError:
        pass
    try:
        ScenarioConfig(churn_hazard=-0.1)
        assert False, "Expected ValueError for churn_hazard < 0"
    except ValueError:
        pass
    # Valid config should not raise
    ScenarioConfig(promo_month=12, churn_hazard=0.0, base_orders_per_month=0.0)


def test_no_transactions_before_acquisition_date() -> None:
    from customer_base_audit.synthetic import generate_transactions
    from customer_base_audit.synthetic.generator import Customer, ScenarioConfig

    cust = Customer(customer_id="X-1", acquisition_date=date(2024, 6, 15))
    txns = generate_transactions(
        [cust],
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        scenario=ScenarioConfig(seed=7, churn_hazard=0.0, base_orders_per_month=2.0),
    )
    assert all(t.event_ts.date() >= date(2024, 6, 1) for t in txns)


def test_high_churn_does_not_precede_acquisition() -> None:
    from customer_base_audit.synthetic import generate_transactions
    from customer_base_audit.synthetic.generator import Customer, ScenarioConfig

    cust = Customer(customer_id="Y-1", acquisition_date=date(2024, 3, 1))
    txns = generate_transactions(
        [cust],
        start=date(2024, 1, 1),
        end=date(2024, 6, 30),
        scenario=ScenarioConfig(seed=11, churn_hazard=1.0, base_orders_per_month=2.0),
    )
    # If churn is immediate, customer may have 0 or some orders but never before acquisition month
    assert all(t.event_ts.date() >= date(2024, 3, 1) for t in txns)
