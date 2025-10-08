"""Cohort assignment utilities for customer segmentation.

This module provides utilities to group customers into cohorts based on
acquisition date, enabling cohort-based analyses in Lenses 3-4. Cohorts
can be automatically generated (monthly, quarterly, yearly) or manually
defined with custom metadata (channel, campaign, region).

Quick Start
-----------
>>> from datetime import datetime
>>> from customer_base_audit.foundation.customer_contract import CustomerIdentifier
>>> from customer_base_audit.foundation.cohorts import create_monthly_cohorts, assign_cohorts
>>>
>>> customers = [
...     CustomerIdentifier("C1", datetime(2023, 1, 15), "crm"),
...     CustomerIdentifier("C2", datetime(2023, 2, 20), "crm"),
... ]
>>> cohorts = create_monthly_cohorts(customers)
>>> assignments = assign_cohorts(customers, cohorts)
>>> print(assignments)
{'C1': '2023-01', 'C2': '2023-02'}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Any, Sequence

from customer_base_audit.foundation.customer_contract import CustomerIdentifier


@dataclass(frozen=True)
class CohortDefinition:
    """Definition of a customer cohort.

    Attributes
    ----------
    cohort_id:
        Unique identifier for the cohort (e.g., "2023-Q1", "paid-search-jan").
    start_date:
        Inclusive start of the acquisition period for this cohort.
    end_date:
        Exclusive end of the acquisition period for this cohort.
    metadata:
        Optional metadata about the cohort (e.g., channel, campaign, region).
    """

    cohort_id: str
    start_date: datetime
    end_date: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cohort definition constraints."""
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date must be before end_date: "
                f"start={self.start_date.isoformat()}, end={self.end_date.isoformat()}"
            )


def _validate_timezone_consistency(customers: Sequence[CustomerIdentifier]) -> None:
    """Validate that all acquisition timestamps have consistent timezone info.

    Parameters
    ----------
    customers:
        List of customer identifiers to validate.

    Raises
    ------
    ValueError
        If customers have mixed timezone-aware and naive datetimes.

    Notes
    -----
    This ensures that cohort date range calculations don't fail due to
    comparing timezone-aware and naive datetimes, which raises TypeError.
    """
    if not customers:
        return

    first_tz = customers[0].acquisition_ts.tzinfo
    for customer in customers[1:]:
        if (customer.acquisition_ts.tzinfo is None) != (first_tz is None):
            raise ValueError(
                f"Mixed timezone-aware and naive datetimes detected. "
                f"All acquisition_ts must be either timezone-aware or naive. "
                f"First customer timezone: {first_tz}, "
                f"problematic customer: {customer.customer_id}"
            )


def assign_cohorts(
    customers: Sequence[CustomerIdentifier],
    cohort_definitions: Sequence[CohortDefinition],
) -> dict[str, str]:
    """Assign customers to cohorts based on acquisition timestamp.

    Parameters
    ----------
    customers:
        List of customer identifiers with acquisition timestamps.
    cohort_definitions:
        List of cohort definitions with acquisition date ranges.

    Returns
    -------
    dict[str, str]
        Mapping of customer_id to cohort_id. Customers whose acquisition_ts
        does not fall within any cohort definition are excluded from the result.

    Raises
    ------
    ValueError
        If duplicate customer_id values are detected in the input.

    Notes
    -----
    - Customers are assigned to the first cohort whose date range contains
      their acquisition_ts (start_date <= acquisition_ts < end_date).
    - If cohort definitions overlap, customers are assigned to the first
      matching cohort in the list order.
    - Customers with acquisition_ts outside all cohort ranges are not assigned.
    - Each customer must appear exactly once; duplicates raise ValueError.
    """
    # Check for duplicate customer IDs
    customer_ids = [c.customer_id for c in customers]
    if len(customer_ids) != len(set(customer_ids)):
        duplicates = [cid for cid in set(customer_ids) if customer_ids.count(cid) > 1]
        raise ValueError(
            f"Duplicate customer_id values detected: {duplicates[:5]}. "
            f"Each customer must appear exactly once."
        )

    assignments: dict[str, str] = {}

    for customer in customers:
        for cohort in cohort_definitions:
            # Check if acquisition falls within cohort date range
            # start_date is inclusive, end_date is exclusive
            if cohort.start_date <= customer.acquisition_ts < cohort.end_date:
                assignments[customer.customer_id] = cohort.cohort_id
                break  # Assign to first matching cohort

    return assignments


def create_monthly_cohorts(
    customers: Sequence[CustomerIdentifier],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[CohortDefinition]:
    """Automatically create monthly acquisition cohorts.

    Parameters
    ----------
    customers:
        List of customer identifiers to determine date range.
    start_date:
        Optional explicit start date for cohort generation. If not provided,
        uses the earliest acquisition_ts from customers.
    end_date:
        Optional explicit end date for cohort generation. If not provided,
        uses the latest acquisition_ts from customers (rounded up to month end).

    Returns
    -------
    list[CohortDefinition]
        List of monthly cohort definitions covering the date range.

    Raises
    ------
    ValueError
        If customers list is empty and start_date/end_date are not provided.

    Examples
    --------
    >>> from datetime import datetime
    >>> customers = [
    ...     CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
    ...     CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
    ... ]
    >>> cohorts = create_monthly_cohorts(customers)
    >>> len(cohorts)
    2
    >>> cohorts[0].cohort_id
    '2023-01'
    """
    if not customers and (start_date is None or end_date is None):
        raise ValueError(
            "Either customers must be non-empty or both start_date and "
            "end_date must be provided"
        )

    # Validate timezone consistency
    _validate_timezone_consistency(customers)

    # Determine date range from customers if not explicitly provided
    if start_date is None or end_date is None:
        acquisition_dates = [c.acquisition_ts for c in customers]
        if start_date is None:
            min_date = min(acquisition_dates)
            # Round down to start of month
            start_date = datetime(
                min_date.year, min_date.month, 1, tzinfo=min_date.tzinfo
            )

        if end_date is None:
            max_date = max(acquisition_dates)
            # Round up to start of next month
            if max_date.month == 12:
                end_date = datetime(max_date.year + 1, 1, 1, tzinfo=max_date.tzinfo)
            else:
                end_date = datetime(
                    max_date.year, max_date.month + 1, 1, tzinfo=max_date.tzinfo
                )

    # Generate monthly cohorts
    cohorts: list[CohortDefinition] = []
    current = start_date

    while current < end_date:
        # Calculate next month boundary
        if current.month == 12:
            next_month = datetime(current.year + 1, 1, 1, tzinfo=current.tzinfo)
        else:
            next_month = datetime(
                current.year, current.month + 1, 1, tzinfo=current.tzinfo
            )

        # Create cohort for this month
        cohort_id = current.strftime("%Y-%m")
        cohorts.append(
            CohortDefinition(
                cohort_id=cohort_id,
                start_date=current,
                end_date=next_month,
                metadata={"year": current.year, "month": current.month},
            )
        )

        current = next_month

    return cohorts


def create_quarterly_cohorts(
    customers: Sequence[CustomerIdentifier],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[CohortDefinition]:
    """Automatically create quarterly acquisition cohorts.

    Parameters
    ----------
    customers:
        List of customer identifiers to determine date range.
    start_date:
        Optional explicit start date for cohort generation. If not provided,
        uses the earliest acquisition_ts from customers.
    end_date:
        Optional explicit end date for cohort generation. If not provided,
        uses the latest acquisition_ts from customers (rounded up to quarter end).

    Returns
    -------
    list[CohortDefinition]
        List of quarterly cohort definitions covering the date range.

    Examples
    --------
    >>> from datetime import datetime
    >>> customers = [
    ...     CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
    ...     CustomerIdentifier("C2", datetime(2023, 5, 20), "system"),
    ... ]
    >>> cohorts = create_quarterly_cohorts(customers)
    >>> len(cohorts)
    2
    >>> cohorts[0].cohort_id
    '2023-Q1'
    """
    if not customers and (start_date is None or end_date is None):
        raise ValueError(
            "Either customers must be non-empty or both start_date and "
            "end_date must be provided"
        )

    # Validate timezone consistency
    _validate_timezone_consistency(customers)

    # Determine date range from customers if not explicitly provided
    if start_date is None or end_date is None:
        acquisition_dates = [c.acquisition_ts for c in customers]
        if start_date is None:
            min_date = min(acquisition_dates)
            # Round down to start of quarter
            quarter_month = ((min_date.month - 1) // 3) * 3 + 1
            start_date = datetime(
                min_date.year, quarter_month, 1, tzinfo=min_date.tzinfo
            )

        if end_date is None:
            max_date = max(acquisition_dates)
            # Round up to start of next quarter
            current_quarter_month = ((max_date.month - 1) // 3) * 3 + 1
            next_quarter_month = current_quarter_month + 3
            if next_quarter_month > 12:
                end_date = datetime(max_date.year + 1, 1, 1, tzinfo=max_date.tzinfo)
            else:
                end_date = datetime(
                    max_date.year, next_quarter_month, 1, tzinfo=max_date.tzinfo
                )

    # Generate quarterly cohorts
    cohorts: list[CohortDefinition] = []
    current = start_date

    while current < end_date:
        # Calculate next quarter boundary
        quarter = (current.month - 1) // 3 + 1
        next_quarter_month = ((quarter % 4) * 3) + 1
        if next_quarter_month == 1:
            next_quarter = datetime(current.year + 1, 1, 1, tzinfo=current.tzinfo)
        else:
            next_quarter = datetime(
                current.year, next_quarter_month, 1, tzinfo=current.tzinfo
            )

        # Create cohort for this quarter
        cohort_id = f"{current.year}-Q{quarter}"
        cohorts.append(
            CohortDefinition(
                cohort_id=cohort_id,
                start_date=current,
                end_date=next_quarter,
                metadata={"year": current.year, "quarter": quarter},
            )
        )

        current = next_quarter

    return cohorts


def create_yearly_cohorts(
    customers: Sequence[CustomerIdentifier],
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> list[CohortDefinition]:
    """Automatically create yearly acquisition cohorts.

    Parameters
    ----------
    customers:
        List of customer identifiers to determine date range.
    start_date:
        Optional explicit start date for cohort generation. If not provided,
        uses the earliest acquisition_ts from customers.
    end_date:
        Optional explicit end date for cohort generation. If not provided,
        uses the latest acquisition_ts from customers (rounded up to year end).

    Returns
    -------
    list[CohortDefinition]
        List of yearly cohort definitions covering the date range.

    Examples
    --------
    >>> from datetime import datetime
    >>> customers = [
    ...     CustomerIdentifier("C1", datetime(2022, 6, 15), "system"),
    ...     CustomerIdentifier("C2", datetime(2023, 5, 20), "system"),
    ... ]
    >>> cohorts = create_yearly_cohorts(customers)
    >>> len(cohorts)
    2
    >>> cohorts[0].cohort_id
    '2022'
    """
    if not customers and (start_date is None or end_date is None):
        raise ValueError(
            "Either customers must be non-empty or both start_date and "
            "end_date must be provided"
        )

    # Validate timezone consistency
    _validate_timezone_consistency(customers)

    # Determine date range from customers if not explicitly provided
    if start_date is None or end_date is None:
        acquisition_dates = [c.acquisition_ts for c in customers]
        if start_date is None:
            min_date = min(acquisition_dates)
            # Round down to start of year
            start_date = datetime(min_date.year, 1, 1, tzinfo=min_date.tzinfo)

        if end_date is None:
            max_date = max(acquisition_dates)
            # Round up to start of next year
            end_date = datetime(max_date.year + 1, 1, 1, tzinfo=max_date.tzinfo)

    # Generate yearly cohorts
    cohorts: list[CohortDefinition] = []
    current = start_date

    while current < end_date:
        # Calculate next year boundary
        next_year = datetime(current.year + 1, 1, 1, tzinfo=current.tzinfo)

        # Create cohort for this year
        cohort_id = str(current.year)
        cohorts.append(
            CohortDefinition(
                cohort_id=cohort_id,
                start_date=current,
                end_date=next_year,
                metadata={"year": current.year},
            )
        )

        current = next_year

    return cohorts


def validate_cohort_coverage(
    customers: Sequence[CustomerIdentifier],
    cohort_definitions: Sequence[CohortDefinition],
) -> tuple[int, int]:
    """Validate cohort coverage and return coverage statistics.

    This utility helps analysts verify that cohort definitions cover all
    customers in the dataset, preventing silent data loss in CLV analysis.

    Parameters
    ----------
    customers:
        List of customer identifiers to check coverage for.
    cohort_definitions:
        List of cohort definitions to validate against.

    Returns
    -------
    tuple[int, int]
        (assigned_count, unassigned_count) where:
        - assigned_count: Number of customers assigned to cohorts
        - unassigned_count: Number of customers outside all cohort ranges

    Examples
    --------
    >>> from datetime import datetime
    >>> customers = [
    ...     CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
    ...     CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
    ...     CustomerIdentifier("C3", datetime(2024, 1, 10), "system"),  # Gap!
    ... ]
    >>> cohorts = [
    ...     CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
    ...     CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
    ... ]
    >>> assigned, unassigned = validate_cohort_coverage(customers, cohorts)
    >>> print(f"Assigned: {assigned}, Unassigned: {unassigned}")
    Assigned: 2, Unassigned: 1
    """
    assignments = assign_cohorts(customers, cohort_definitions)
    assigned = len(assignments)
    unassigned = len(customers) - assigned
    return assigned, unassigned
