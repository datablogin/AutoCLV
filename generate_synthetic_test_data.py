#!/usr/bin/env python
"""
Generate synthetic transaction data for testing Four Lenses MCP tools.

This script creates a JSON file with synthetic customer transactions that can be
used to test the MCP server with Claude Desktop.

Usage:
    python generate_synthetic_test_data.py

Output:
    synthetic_transactions.json - Transaction data for MCP testing
"""

import json
from datetime import date
from pathlib import Path

from customer_base_audit.synthetic import (
    generate_customers,
    generate_transactions,
    BASELINE_SCENARIO,
)


def main():
    """Generate synthetic transaction data and save to JSON."""
    print("Generating synthetic customer data...")

    # Generate 4000 customers over 1 year for robust cohort analysis
    customers = generate_customers(
        n=4000,
        start=date(2023, 1, 1),
        end=date(2023, 12, 31),
        seed=42  # Fixed seed for reproducibility
    )

    print(f"Generated {len(customers)} customers")

    # Generate transaction line items
    print("Generating transaction line items...")
    line_items = generate_transactions(
        customers=customers,
        start=date(2023, 1, 1),
        end=date(2024, 6, 30),  # 18 months of data
        scenario=BASELINE_SCENARIO
    )

    print(f"Generated {len(line_items)} line items")

    # Convert line items to transaction format expected by CustomerDataMartBuilder
    print("Converting to transaction format...")
    transactions = []
    from datetime import timezone

    for item in line_items:
        # Add timezone info (UTC) to timestamps
        event_ts = item.event_ts
        if event_ts.tzinfo is None:
            event_ts = event_ts.replace(tzinfo=timezone.utc)

        # CustomerDataMartBuilder expects these fields:
        # - customer_id, order_id
        # - event_ts or order_ts (datetime with timezone)
        # - unit_price, quantity
        # - product_id (optional)
        # It will calculate line_total = unit_price * quantity
        transactions.append({
            'customer_id': item.customer_id,
            'order_id': item.order_id,
            'event_ts': event_ts.isoformat(),  # Now includes timezone
            'unit_price': float(item.unit_price),
            'quantity': item.quantity,
            'product_id': item.product_id,
            # Optional: Add margin data if needed
            'unit_cost': float(item.unit_price * 0.7),  # Assume 30% margin
        })

    # Sort by timestamp
    transactions.sort(key=lambda x: x['event_ts'])

    print(f"Created {len(transactions)} transaction records")

    # Save to JSON
    output_file = Path("synthetic_transactions.json")

    with open(output_file, 'w') as f:
        json.dump(transactions, f, indent=2)

    print(f"\n✅ Synthetic data saved to: {output_file.absolute()}")
    print("\nTransaction statistics:")
    print(f"  - Total line items: {len(transactions)}")
    print(f"  - Unique orders: {len(set(t['order_id'] for t in transactions))}")
    print(f"  - Unique customers: {len(set(t['customer_id'] for t in transactions))}")
    print(f"  - Date range: {transactions[0]['event_ts'][:10]} to {transactions[-1]['event_ts'][:10]}")
    print(f"  - Total revenue: ${sum(t['unit_price'] * t['quantity'] for t in transactions):,.2f}")
    print("\nYou can now test the MCP server with this data!")
    print(f"File path: {output_file.absolute()}")


if __name__ == "__main__":
    main()
