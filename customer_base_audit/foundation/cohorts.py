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

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Any, Sequence, TypedDict

from customer_base_audit.foundation.customer_contract import CustomerIdentifier

logger = logging.getLogger(__name__)


class CohortMetadata(TypedDict, total=False):
    """Common cohort metadata fields.

    All fields are optional. This TypedDict provides type safety and IDE
    auto-completion for common metadata patterns.

    Attributes
    ----------
    description:
        Human-readable description of the cohort
    campaign_id:
        Marketing campaign identifier
    acquisition_channel:
        Channel through which customers were acquired (e.g., "paid-search", "organic", "email")
    created_by:
        User or system that created the cohort
    created_at:
        ISO 8601 timestamp when the cohort was created
    """

    description: str
    campaign_id: str
    acquisition_channel: str
    created_by: str
    created_at: str


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
        Optional metadata about the cohort. Use CohortMetadata TypedDict for
        type-safe common fields (description, campaign_id, acquisition_channel, etc.),
        or Mapping[str, Any] for custom fields.
    """

    cohort_id: str
    start_date: datetime
    end_date: datetime
    metadata: CohortMetadata | Mapping[str, Any] = field(default_factory=dict)

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
        If customers have mixed timezone-aware and naive datetimes, or if
        timezone-aware customers use different timezones.

    Notes
    -----
    This ensures that cohort date range calculations don't fail due to
    comparing timezone-aware and naive datetimes (raises TypeError), or
    comparing different timezones which could lead to incorrect cohort
    assignments (e.g., UTC vs US/Eastern).
    """
    if not customers:
        return

    first_tz = customers[0].acquisition_ts.tzinfo
    for customer in customers:
        if customer.acquisition_ts.tzinfo != first_tz:
            raise ValueError(
                f"Inconsistent timezone detected. "
                f"Expected {first_tz}, got {customer.acquisition_ts.tzinfo} "
                f"for customer {customer.customer_id}. "
                f"All acquisition_ts must use the same timezone."
            )


def normalize_to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC timezone.

    This utility function helps prevent timezone-related bugs in distributed
    analytics systems by enforcing consistent UTC timestamps.

    Parameters
    ----------
    dt:
        Datetime to normalize to UTC

    Returns
    -------
    datetime
        Datetime converted to UTC timezone

    Raises
    ------
    ValueError
        If the datetime is naive (no timezone info). All datetimes must be
        timezone-aware for safe conversion.

    Examples
    --------
    >>> from datetime import datetime, timezone, timedelta
    >>> from customer_base_audit.foundation.cohorts import normalize_to_utc
    >>>
    >>> # Convert timezone-aware datetime to UTC
    >>> dt_eastern = datetime(2023, 1, 15, 10, 0, tzinfo=timezone(timedelta(hours=-5)))
    >>> dt_utc = normalize_to_utc(dt_eastern)
    >>> dt_utc.tzinfo
    datetime.timezone.utc
    >>> dt_utc.hour
    15

    Notes
    -----
    **Best Practice for CLV Analytics**: Always use UTC timestamps for all
    customer analytics to avoid timezone-related bugs, especially when:
    - Aggregating data from multiple regions
    - Comparing metrics across different time periods
    - Integrating with distributed systems

    This aligns with the Customer-Base Audit framework's requirement for
    consistent timezone handling across all Five Lenses analyses.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "Naive datetime not allowed. Use timezone-aware datetimes. "
            "For UTC timestamps, use: datetime(..., tzinfo=timezone.utc)"
        )
    return dt.astimezone(timezone.utc)


def validate_non_overlapping(cohort_definitions: Sequence[CohortDefinition]) -> None:
    """Validate that cohort definitions do not overlap.

    Parameters
    ----------
    cohort_definitions:
        List of cohort definitions to check for overlaps.

    Raises
    ------
    ValueError
        If any cohorts have overlapping date ranges.

    Notes
    -----
    This is useful for custom cohort definitions where overlaps might
    be unintentional. The standard `assign_cohorts()` behavior is to
    assign to the first matching cohort, but overlaps could indicate
    a configuration error.

    Examples
    --------
    >>> from datetime import datetime
    >>> cohorts = [
    ...     CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
    ...     CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
    ... ]
    >>> validate_non_overlapping(cohorts)  # No overlap, passes
    >>> overlapping = [
    ...     CohortDefinition("A", datetime(2023, 1, 1), datetime(2023, 2, 15)),
    ...     CohortDefinition("B", datetime(2023, 2, 1), datetime(2023, 3, 1)),
    ... ]
    >>> validate_non_overlapping(overlapping)  # doctest: +SKIP
    ValueError: Overlapping cohorts detected: A and B
    """
    if not cohort_definitions:
        return

    sorted_cohorts = sorted(cohort_definitions, key=lambda c: c.start_date)

    for i in range(len(sorted_cohorts) - 1):
        current = sorted_cohorts[i]
        next_cohort = sorted_cohorts[i + 1]

        if current.end_date > next_cohort.start_date:
            raise ValueError(
                f"Overlapping cohorts detected: '{current.cohort_id}' "
                f"(ends {current.end_date.isoformat()}) overlaps with "
                f"'{next_cohort.cohort_id}' (starts {next_cohort.start_date.isoformat()})"
            )


def assign_cohorts(
    customers: Sequence[CustomerIdentifier],
    cohort_definitions: Sequence[CohortDefinition],
    require_full_coverage: bool = True,
) -> dict[str, str]:
    """Assign customers to cohorts based on acquisition timestamp.

    Parameters
    ----------
    customers:
        List of customer identifiers with acquisition timestamps.
    cohort_definitions:
        List of cohort definitions with acquisition date ranges.
    require_full_coverage:
        If True, raise ValueError if any customers fall outside cohort ranges.
        This prevents silent data loss in CLV analysis. Default True (changed
        from False in v2.0 for data safety). Set to False to allow partial
        coverage with warning logging instead.

    Returns
    -------
    dict[str, str]
        Mapping of customer_id to cohort_id. Customers whose acquisition_ts
        does not fall within any cohort definition are excluded from the result
        (unless require_full_coverage=True, which raises an error instead).

    Raises
    ------
    ValueError
        If duplicate customer_id values are detected in the input, or if
        require_full_coverage=True and any customers fall outside cohort ranges.

    Notes
    -----
    - Customers are assigned to the first cohort whose date range contains
      their acquisition_ts (start_date <= acquisition_ts < end_date).
    - If cohort definitions overlap, customers are assigned to the first
      matching cohort in the list order.
    - Customers with acquisition_ts outside all cohort ranges are not assigned
      (or raise error if require_full_coverage=True).
    - Each customer must appear exactly once; duplicates raise ValueError.
    - **Data Safety**: Default changed to require_full_coverage=True in v2.0
      to prevent silent data loss. If you need partial coverage, explicitly set
      require_full_coverage=False and review warning logs.

    Examples
    --------
    >>> customers = [CustomerIdentifier("C1", datetime(2023, 1, 15), "sys")]
    >>> cohorts = [CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1))]
    >>> assign_cohorts(customers, cohorts)
    {'C1': '2023-01'}
    >>> # With strict coverage validation (default behavior):
    >>> assign_cohorts(customers, [], require_full_coverage=True)  # doctest: +SKIP
    ValueError: 1 customers fall outside cohort ranges
    >>> # Allow partial coverage with warnings (opt-in):
    >>> assign_cohorts(customers, cohorts, require_full_coverage=False)  # Logs warnings
    {'C1': '2023-01'}
    """
    # Check for duplicate customer IDs (O(n) using Counter)
    customer_ids = [c.customer_id for c in customers]
    id_counts = Counter(customer_ids)
    duplicates = [cid for cid, count in id_counts.items() if count > 1]
    if duplicates:
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

    # Check for unassigned customers
    if len(assignments) < len(customers):
        unassigned = [
            c.customer_id for c in customers if c.customer_id not in assignments
        ]
        unassigned_count = len(unassigned)
        coverage_pct = (len(assignments) / len(customers)) * 100 if customers else 100

        if require_full_coverage:
            # Strict mode: raise error on any data loss
            raise ValueError(
                f"{unassigned_count} customers fall outside cohort ranges "
                f"({coverage_pct:.1f}% coverage). "
                f"First 5 unassigned: {unassigned[:5]}. "
                f"This may indicate missing cohort definitions or data quality issues. "
                f"To allow partial coverage, set require_full_coverage=False (not recommended)."
            )
        else:
            # Permissive mode: log warning but continue
            logger.warning(
                f"Cohort assignment incomplete: {unassigned_count}/{len(customers)} customers "
                f"({(unassigned_count/len(customers))*100:.1f}%) fall outside cohort ranges "
                f"({coverage_pct:.1f}% coverage). "
                f"These customers will be excluded from cohort analysis. "
                f"First 5 unassigned: {unassigned[:5]}. "
                f"Consider expanding cohort definitions or setting require_full_coverage=True "
                f"to catch data quality issues."
            )

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
    # Allow partial coverage to get statistics (validation purpose)
    assignments = assign_cohorts(
        customers, cohort_definitions, require_full_coverage=False
    )
    assigned = len(assignments)
    unassigned = len(customers) - assigned
    return assigned, unassigned
