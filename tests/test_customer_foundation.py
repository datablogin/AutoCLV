from datetime import datetime

import pytest

from customer_base_audit.foundation import (
    CustomerContract,
    CustomerDataMartBuilder,
    PeriodGranularity,
)


def test_customer_contract_merge_and_validation():
    contract = CustomerContract()
    identifiers = contract.validate_records(
        [
            {
                "customer_id": "123",
                "acquisition_ts": datetime(2024, 1, 1, 12, 0),
                "metadata": {"loyalty": "gold"},
            },
            {
                "customer_id": "123",
                "acquisition_ts": datetime(2024, 2, 1, 9, 0),
                "is_visible": False,
                "metadata": {"region": "emea"},
            },
            {
                "customer_id": "456",
                "acquisition_ts": datetime(2023, 7, 10, 8, 30),
            },
        ],
        source_system="crm",
    )
    merged = contract.merge(identifiers)
    merged.sort(key=lambda item: item.customer_id)

    assert merged[0].customer_id == "123"
    assert merged[0].acquisition_ts == datetime(2024, 1, 1, 12, 0)
    assert merged[0].is_visible is True  # one record was visible
    assert merged[0].metadata == {"loyalty": "gold", "region": "emea"}

    assert merged[1].customer_id == "456"
    assert merged[1].acquisition_ts == datetime(2023, 7, 10, 8, 30)


def test_customer_contract_filters_invisible() -> None:
    contract = CustomerContract(enforce_visibility=True)
    identifiers = contract.validate_records(
        [
            {
                "customer_id": "999",
                "acquisition_ts": datetime(2023, 3, 3),
                "is_visible": False,
            }
        ],
        source_system="crm",
    )
    assert identifiers == []


def test_customer_contract_rejects_missing_fields():
    contract = CustomerContract()
    with pytest.raises(ValueError):
        contract.validate_records([{"customer_id": "1"}], source_system="crm")


def test_customer_contract_rejects_out_of_range_timestamp():
    contract = CustomerContract()
    with pytest.raises(ValueError):
        contract.validate_records(
            [
                {
                    "customer_id": "666",
                    "acquisition_ts": datetime(1900, 1, 1),
                }
            ],
            source_system="crm",
        )


@pytest.fixture
def sample_transactions():
    return [
        {
            "order_id": "A-1",
            "customer_id": "123",
            "event_ts": datetime(2024, 1, 4, 10, 2),
            "product_id": "SKU-1",
            "quantity": 2,
            "unit_price": 30.0,
        },
        {
            "order_id": "A-1",
            "customer_id": "123",
            "event_ts": datetime(2024, 1, 4, 10, 5),
            "product_id": "SKU-2",
            "quantity": 1,
            "unit_price": 15.0,
            "unit_cost": 10.0,
        },
        {
            "order_id": "B-1",
            "customer_id": "456",
            "event_ts": datetime(2024, 4, 12, 15, 20),
            "product_id": "SKU-3",
            "quantity": 3,
            "unit_price": 10.0,
        },
        {
            "order_id": "C-1",
            "customer_id": "789",
            "event_ts": datetime(2024, 12, 15, 9, 0),
            "product_id": "SKU-4",
            "quantity": 1,
            "unit_price": 50.0,
        },
    ]


def test_data_mart_builder_orders(sample_transactions):
    builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.QUARTER])
    mart = builder.build(sample_transactions)

    assert len(mart.orders) == 3
    order = mart.orders[0]
    assert order.order_id == "A-1"
    assert order.customer_id == "123"
    assert order.total_spend == 75.0
    assert order.total_margin == pytest.approx(65.0)
    assert order.total_quantity == 3
    assert order.distinct_products == 2


def test_data_mart_builder_periods(sample_transactions):
    builder = CustomerDataMartBuilder(period_granularities=[PeriodGranularity.QUARTER])
    mart = builder.build(sample_transactions)

    periods = mart.periods[PeriodGranularity.QUARTER]
    assert len(periods) == 3
    first = periods[0]
    assert first.customer_id == "123"
    assert first.total_orders == 1
    assert first.total_spend == 75.0
    assert first.total_margin == pytest.approx(65.0)
    assert first.total_quantity == 3

    second = periods[1]
    assert second.customer_id == "456"
    assert second.total_orders == 1
    assert second.total_spend == 30.0
    assert second.total_margin == pytest.approx(30.0)

    third = periods[2]
    assert third.customer_id == "789"
    assert third.period_end == datetime(2025, 1, 1)


def test_data_mart_builder_multiple_periods(sample_transactions):
    builder = CustomerDataMartBuilder(
        period_granularities=[PeriodGranularity.QUARTER, PeriodGranularity.YEAR]
    )
    mart = builder.build(sample_transactions)

    assert set(mart.periods.keys()) == {
        PeriodGranularity.QUARTER,
        PeriodGranularity.YEAR,
    }

    yearly = mart.periods[PeriodGranularity.YEAR]
    assert len(yearly) == 3
    totals = {aggregate.customer_id: aggregate for aggregate in yearly}
    assert totals["123"].total_spend == 75.0
    assert totals["456"].total_spend == 30.0
    assert totals["789"].total_spend == 50.0


def test_aggregate_orders_rejects_negative_price():
    builder = CustomerDataMartBuilder()
    with pytest.raises(ValueError):
        builder.build(
            [
                {
                    "order_id": "NEG-1",
                    "customer_id": "321",
                    "event_ts": datetime(2024, 5, 1, 12, 0),
                    "quantity": 1,
                    "unit_price": -10,
                }
            ]
        )
