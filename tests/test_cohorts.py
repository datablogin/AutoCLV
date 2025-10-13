"""Unit tests for cohort assignment infrastructure."""

from datetime import datetime

import pytest

from customer_base_audit.foundation.customer_contract import CustomerIdentifier
from customer_base_audit.foundation.cohorts import (
    CohortDefinition,
    assign_cohorts,
    create_monthly_cohorts,
    create_quarterly_cohorts,
    create_yearly_cohorts,
)


class TestCohortDefinition:
    """Test CohortDefinition dataclass validation."""

    def test_valid_cohort_definition(self):
        """Test creating a valid cohort definition."""
        cohort = CohortDefinition(
            cohort_id="2023-Q1",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 4, 1),
            metadata={"channel": "paid_search"},
        )
        assert cohort.cohort_id == "2023-Q1"
        assert cohort.start_date == datetime(2023, 1, 1)
        assert cohort.end_date == datetime(2023, 4, 1)
        assert cohort.metadata == {"channel": "paid_search"}

    def test_cohort_definition_with_default_metadata(self):
        """Test cohort definition without explicit metadata."""
        cohort = CohortDefinition(
            cohort_id="2023-01",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1),
        )
        assert cohort.metadata == {}

    def test_cohort_definition_validation_fails_when_start_after_end(self):
        """Test validation fails when start_date >= end_date."""
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            CohortDefinition(
                cohort_id="invalid",
                start_date=datetime(2023, 2, 1),
                end_date=datetime(2023, 1, 1),
            )

    def test_cohort_definition_validation_fails_when_start_equals_end(self):
        """Test validation fails when start_date == end_date."""
        same_date = datetime(2023, 1, 1)
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            CohortDefinition(
                cohort_id="invalid",
                start_date=same_date,
                end_date=same_date,
            )


class TestAssignCohorts:
    """Test cohort assignment logic."""

    def test_assign_customers_to_single_cohort(self):
        """Test assigning all customers to a single cohort."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 20), "system"),
            CustomerIdentifier("C3", datetime(2023, 1, 31), "system"),
        ]
        cohorts = [
            CohortDefinition(
                cohort_id="2023-01",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 2, 1),
            )
        ]

        assignments = assign_cohorts(customers, cohorts)

        assert len(assignments) == 3
        assert assignments["C1"] == "2023-01"
        assert assignments["C2"] == "2023-01"
        assert assignments["C3"] == "2023-01"

    def test_assign_customers_to_multiple_cohorts(self):
        """Test assigning customers to different cohorts."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
            CustomerIdentifier("C3", datetime(2023, 3, 10), "system"),
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
            CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
            CohortDefinition("2023-03", datetime(2023, 3, 1), datetime(2023, 4, 1)),
        ]

        assignments = assign_cohorts(customers, cohorts)

        assert len(assignments) == 3
        assert assignments["C1"] == "2023-01"
        assert assignments["C2"] == "2023-02"
        assert assignments["C3"] == "2023-03"

    def test_customers_on_cohort_boundaries(self):
        """Test boundary conditions: start_date inclusive, end_date exclusive."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 1, 0, 0, 0), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 1, 0, 0, 0), "system"),
            CustomerIdentifier("C3", datetime(2023, 1, 31, 23, 59, 59), "system"),
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
            CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
        ]

        assignments = assign_cohorts(customers, cohorts)

        assert assignments["C1"] == "2023-01"  # On start boundary: included
        assert assignments["C2"] == "2023-02"  # On end boundary: goes to next cohort
        assert assignments["C3"] == "2023-01"  # Just before end: included

    def test_customers_outside_all_cohorts_raises_error_by_default(self):
        """Test customers with acquisition_ts outside all cohorts raise error (default behavior)."""
        customers = [
            CustomerIdentifier("C1", datetime(2022, 12, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C3", datetime(2023, 4, 15), "system"),
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
            CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
            CohortDefinition("2023-03", datetime(2023, 3, 1), datetime(2023, 4, 1)),
        ]

        # Default behavior (require_full_coverage=True) should raise error
        with pytest.raises(ValueError, match="2 customers fall outside cohort ranges"):
            assign_cohorts(customers, cohorts)

    def test_customers_outside_all_cohorts_not_assigned_when_allowed(self):
        """Test customers outside cohorts are excluded when require_full_coverage=False."""
        customers = [
            CustomerIdentifier("C1", datetime(2022, 12, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C3", datetime(2023, 4, 15), "system"),
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
            CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
            CohortDefinition("2023-03", datetime(2023, 3, 1), datetime(2023, 4, 1)),
        ]

        # Explicitly allow partial coverage
        assignments = assign_cohorts(customers, cohorts, require_full_coverage=False)

        assert len(assignments) == 1  # Only C2 is assigned
        assert assignments["C2"] == "2023-01"
        assert "C1" not in assignments  # Too early
        assert "C3" not in assignments  # Too late

    def test_overlapping_cohorts_assigns_to_first_match(self):
        """Test that overlapping cohorts assign customers to first matching cohort."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
        ]
        cohorts = [
            CohortDefinition("all-jan", datetime(2023, 1, 1), datetime(2023, 2, 1)),
            CohortDefinition("mid-jan", datetime(2023, 1, 10), datetime(2023, 1, 20)),
        ]

        assignments = assign_cohorts(customers, cohorts)

        assert assignments["C1"] == "all-jan"  # First matching cohort

    def test_empty_customers_list(self):
        """Test assigning empty customers list."""
        customers = []
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
        ]

        assignments = assign_cohorts(customers, cohorts)

        assert assignments == {}

    def test_empty_cohorts_list_raises_error_by_default(self):
        """Test assigning with empty cohorts list raises error (default behavior)."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
        ]
        cohorts = []

        # Default behavior should raise error
        with pytest.raises(ValueError, match="1 customers fall outside cohort ranges"):
            assign_cohorts(customers, cohorts)

    def test_empty_cohorts_list_allowed_with_flag(self):
        """Test assigning with empty cohorts list when allowed."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
        ]
        cohorts = []

        # Explicitly allow partial coverage
        assignments = assign_cohorts(customers, cohorts, require_full_coverage=False)

        assert assignments == {}

    def test_cohort_sizes_sum_to_total_customer_count(self):
        """Test that all customers are assigned when cohorts cover full range."""
        customers = [
            CustomerIdentifier(f"C{i}", datetime(2023, 1, i), "system")
            for i in range(1, 32)  # 31 customers across January
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
        ]

        assignments = assign_cohorts(customers, cohorts)

        assert len(assignments) == 31
        assert all(v == "2023-01" for v in assignments.values())


class TestCreateMonthlyCohorts:
    """Test automatic monthly cohort generation."""

    def test_create_monthly_cohorts_from_customers(self):
        """Test generating monthly cohorts from customer acquisition dates."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
            CustomerIdentifier("C3", datetime(2023, 3, 10), "system"),
        ]

        cohorts = create_monthly_cohorts(customers)

        assert len(cohorts) == 3
        assert cohorts[0].cohort_id == "2023-01"
        assert cohorts[0].start_date == datetime(2023, 1, 1)
        assert cohorts[0].end_date == datetime(2023, 2, 1)

        assert cohorts[1].cohort_id == "2023-02"
        assert cohorts[1].start_date == datetime(2023, 2, 1)
        assert cohorts[1].end_date == datetime(2023, 3, 1)

        assert cohorts[2].cohort_id == "2023-03"
        assert cohorts[2].start_date == datetime(2023, 3, 1)
        assert cohorts[2].end_date == datetime(2023, 4, 1)

    def test_create_monthly_cohorts_with_explicit_dates(self):
        """Test generating monthly cohorts with explicit start/end dates."""
        customers = []
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 4, 1)

        cohorts = create_monthly_cohorts(customers, start_date, end_date)

        assert len(cohorts) == 3
        assert cohorts[0].cohort_id == "2023-01"
        assert cohorts[1].cohort_id == "2023-02"
        assert cohorts[2].cohort_id == "2023-03"

    def test_create_monthly_cohorts_spans_year_boundary(self):
        """Test monthly cohorts spanning December to January."""
        customers = [
            CustomerIdentifier("C1", datetime(2022, 12, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 20), "system"),
        ]

        cohorts = create_monthly_cohorts(customers)

        assert len(cohorts) == 2
        assert cohorts[0].cohort_id == "2022-12"
        assert cohorts[0].start_date == datetime(2022, 12, 1)
        assert cohorts[0].end_date == datetime(2023, 1, 1)

        assert cohorts[1].cohort_id == "2023-01"
        assert cohorts[1].start_date == datetime(2023, 1, 1)
        assert cohorts[1].end_date == datetime(2023, 2, 1)

    def test_create_monthly_cohorts_single_month(self):
        """Test creating cohorts when all customers in same month."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 5), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C3", datetime(2023, 1, 25), "system"),
        ]

        cohorts = create_monthly_cohorts(customers)

        assert len(cohorts) == 1
        assert cohorts[0].cohort_id == "2023-01"

    def test_create_monthly_cohorts_includes_metadata(self):
        """Test that monthly cohorts include year and month metadata."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 3, 15), "system"),
        ]

        cohorts = create_monthly_cohorts(customers)

        assert cohorts[0].metadata["year"] == 2023
        assert cohorts[0].metadata["month"] == 3

    def test_create_monthly_cohorts_empty_customers_without_dates_raises(self):
        """Test that empty customers without explicit dates raises error."""
        with pytest.raises(ValueError, match="Either customers must be non-empty"):
            create_monthly_cohorts([])


class TestCreateQuarterlyCohorts:
    """Test automatic quarterly cohort generation."""

    def test_create_quarterly_cohorts_from_customers(self):
        """Test generating quarterly cohorts from customer acquisition dates."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 5, 20), "system"),
        ]

        cohorts = create_quarterly_cohorts(customers)

        assert len(cohorts) == 2
        assert cohorts[0].cohort_id == "2023-Q1"
        assert cohorts[0].start_date == datetime(2023, 1, 1)
        assert cohorts[0].end_date == datetime(2023, 4, 1)

        assert cohorts[1].cohort_id == "2023-Q2"
        assert cohorts[1].start_date == datetime(2023, 4, 1)
        assert cohorts[1].end_date == datetime(2023, 7, 1)

    def test_create_quarterly_cohorts_spans_year_boundary(self):
        """Test quarterly cohorts spanning Q4 to Q1."""
        customers = [
            CustomerIdentifier("C1", datetime(2022, 11, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
        ]

        cohorts = create_quarterly_cohorts(customers)

        assert len(cohorts) == 2
        assert cohorts[0].cohort_id == "2022-Q4"
        assert cohorts[0].start_date == datetime(2022, 10, 1)
        assert cohorts[0].end_date == datetime(2023, 1, 1)

        assert cohorts[1].cohort_id == "2023-Q1"
        assert cohorts[1].start_date == datetime(2023, 1, 1)
        assert cohorts[1].end_date == datetime(2023, 4, 1)

    def test_create_quarterly_cohorts_includes_metadata(self):
        """Test that quarterly cohorts include year and quarter metadata."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 8, 15), "system"),
        ]

        cohorts = create_quarterly_cohorts(customers)

        assert cohorts[0].cohort_id == "2023-Q3"
        assert cohorts[0].metadata["year"] == 2023
        assert cohorts[0].metadata["quarter"] == 3

    def test_create_quarterly_cohorts_with_explicit_dates(self):
        """Test generating quarterly cohorts with explicit start/end dates."""
        customers = []
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 1, 1)

        cohorts = create_quarterly_cohorts(customers, start_date, end_date)

        assert len(cohorts) == 4
        assert cohorts[0].cohort_id == "2023-Q1"
        assert cohorts[1].cohort_id == "2023-Q2"
        assert cohorts[2].cohort_id == "2023-Q3"
        assert cohorts[3].cohort_id == "2023-Q4"


class TestCreateYearlyCohorts:
    """Test automatic yearly cohort generation."""

    def test_create_yearly_cohorts_from_customers(self):
        """Test generating yearly cohorts from customer acquisition dates."""
        customers = [
            CustomerIdentifier("C1", datetime(2022, 6, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 5, 20), "system"),
        ]

        cohorts = create_yearly_cohorts(customers)

        assert len(cohorts) == 2
        assert cohorts[0].cohort_id == "2022"
        assert cohorts[0].start_date == datetime(2022, 1, 1)
        assert cohorts[0].end_date == datetime(2023, 1, 1)

        assert cohorts[1].cohort_id == "2023"
        assert cohorts[1].start_date == datetime(2023, 1, 1)
        assert cohorts[1].end_date == datetime(2024, 1, 1)

    def test_create_yearly_cohorts_single_year(self):
        """Test creating cohorts when all customers in same year."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 5), "system"),
            CustomerIdentifier("C2", datetime(2023, 6, 15), "system"),
            CustomerIdentifier("C3", datetime(2023, 12, 25), "system"),
        ]

        cohorts = create_yearly_cohorts(customers)

        assert len(cohorts) == 1
        assert cohorts[0].cohort_id == "2023"
        assert cohorts[0].start_date == datetime(2023, 1, 1)
        assert cohorts[0].end_date == datetime(2024, 1, 1)

    def test_create_yearly_cohorts_includes_metadata(self):
        """Test that yearly cohorts include year metadata."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 8, 15), "system"),
        ]

        cohorts = create_yearly_cohorts(customers)

        assert cohorts[0].metadata["year"] == 2023

    def test_create_yearly_cohorts_with_explicit_dates(self):
        """Test generating yearly cohorts with explicit start/end dates."""
        customers = []
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 1, 1)

        cohorts = create_yearly_cohorts(customers, start_date, end_date)

        assert len(cohorts) == 3
        assert cohorts[0].cohort_id == "2020"
        assert cohorts[1].cohort_id == "2021"
        assert cohorts[2].cohort_id == "2022"


class TestCohortIntegration:
    """Integration tests combining cohort creation and assignment."""

    def test_monthly_cohort_creation_and_assignment(self):
        """Test end-to-end: create monthly cohorts and assign customers."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
            CustomerIdentifier("C3", datetime(2023, 3, 10), "system"),
        ]

        cohorts = create_monthly_cohorts(customers)
        assignments = assign_cohorts(customers, cohorts)

        assert len(assignments) == 3
        assert assignments["C1"] == "2023-01"
        assert assignments["C2"] == "2023-02"
        assert assignments["C3"] == "2023-03"

    def test_quarterly_cohort_creation_and_assignment(self):
        """Test end-to-end: create quarterly cohorts and assign customers."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 5, 20), "system"),
            CustomerIdentifier("C3", datetime(2023, 10, 10), "system"),
        ]

        cohorts = create_quarterly_cohorts(customers)
        assignments = assign_cohorts(customers, cohorts)

        assert len(assignments) == 3
        assert assignments["C1"] == "2023-Q1"
        assert assignments["C2"] == "2023-Q2"
        assert assignments["C3"] == "2023-Q4"

    def test_custom_cohorts_with_metadata(self):
        """Test custom cohort definitions with metadata."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 20), "system"),
        ]

        cohorts = [
            CohortDefinition(
                cohort_id="paid-search-jan",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 2, 1),
                metadata={"channel": "paid_search", "campaign": "winter_promo"},
            )
        ]

        assignments = assign_cohorts(customers, cohorts)

        assert len(assignments) == 2
        assert assignments["C1"] == "paid-search-jan"
        assert assignments["C2"] == "paid-search-jan"
        assert cohorts[0].metadata["channel"] == "paid_search"
        assert cohorts[0].metadata["campaign"] == "winter_promo"


class TestTimezoneHandling:
    """Test timezone-aware datetime handling."""

    def test_timezone_aware_datetimes(self):
        """Test cohort assignment with timezone-aware datetimes."""
        from datetime import timezone

        customers = [
            CustomerIdentifier(
                "C1", datetime(2023, 1, 15, tzinfo=timezone.utc), "system"
            ),
            CustomerIdentifier(
                "C2", datetime(2023, 2, 20, tzinfo=timezone.utc), "system"
            ),
        ]

        cohorts = create_monthly_cohorts(customers)

        assert len(cohorts) == 2
        assert cohorts[0].start_date.tzinfo == timezone.utc
        assert cohorts[0].end_date.tzinfo == timezone.utc
        assert cohorts[1].start_date.tzinfo == timezone.utc
        assert cohorts[1].end_date.tzinfo == timezone.utc

    def test_mixed_timezone_raises_error(self):
        """Test that mixed timezone-aware and naive datetimes raise error."""
        from datetime import timezone

        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),  # Naive
            CustomerIdentifier(
                "C2", datetime(2023, 2, 20, tzinfo=timezone.utc), "system"
            ),  # Aware
        ]

        with pytest.raises(ValueError, match="Inconsistent timezone"):
            create_monthly_cohorts(customers)

    def test_different_timezone_aware_raises_error(self):
        """Test that different timezone-aware datetimes raise error."""
        from datetime import timezone
        from zoneinfo import ZoneInfo

        customers = [
            CustomerIdentifier(
                "C1", datetime(2023, 1, 15, tzinfo=timezone.utc), "system"
            ),  # UTC
            CustomerIdentifier(
                "C2",
                datetime(2023, 2, 20, tzinfo=ZoneInfo("US/Eastern")),
                "system",
            ),  # Eastern
        ]

        with pytest.raises(ValueError, match="Inconsistent timezone"):
            create_monthly_cohorts(customers)


class TestDataQuality:
    """Test data quality and validation features."""

    def test_duplicate_customer_ids_raises_error(self):
        """Test that duplicate customer IDs raise an error."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C1", datetime(2023, 1, 20), "system"),  # Duplicate
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1))
        ]

        with pytest.raises(ValueError, match="Duplicate customer_id"):
            assign_cohorts(customers, cohorts)

    def test_duplicate_detection_shows_ids(self):
        """Test that duplicate detection shows which IDs are duplicated."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C1", datetime(2023, 1, 20), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 25), "system"),
            CustomerIdentifier("C2", datetime(2023, 1, 30), "system"),
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1))
        ]

        with pytest.raises(ValueError) as exc_info:
            assign_cohorts(customers, cohorts)

        error_msg = str(exc_info.value)
        assert "C1" in error_msg or "C2" in error_msg

    def test_empty_cohort_definitions_raises_error_by_default(self):
        """Test assignment with empty cohort list raises error (default behavior)."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
        ]

        # Default behavior should raise error
        with pytest.raises(ValueError, match="1 customers fall outside cohort ranges"):
            assign_cohorts(customers, [])

    def test_empty_cohort_definitions_allowed_with_flag(self):
        """Test assignment with empty cohort list when explicitly allowed."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
        ]

        # Explicitly allow partial coverage
        assignments = assign_cohorts(customers, [], require_full_coverage=False)

        assert assignments == {}

    def test_require_full_coverage_raises_on_gaps(self):
        """Test that require_full_coverage raises error for unassigned customers."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 3, 20), "system"),  # Gap!
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
        ]

        with pytest.raises(ValueError, match="1 customers fall outside cohort ranges"):
            assign_cohorts(customers, cohorts, require_full_coverage=True)

    def test_partial_coverage_logs_warning_when_allowed(self, caplog):
        """Test that partial coverage logs warning when require_full_coverage=False."""
        import logging

        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 3, 20), "system"),  # Outside cohort!
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
        ]

        with caplog.at_level(logging.WARNING):
            assignments = assign_cohorts(customers, cohorts, require_full_coverage=False)

        # Should assign only C1
        assert len(assignments) == 1
        assert assignments["C1"] == "2023-01"
        assert "C2" not in assignments

        # Should log warning about C2
        assert any("Cohort assignment incomplete" in record.message for record in caplog.records)
        assert any("1/2 customers" in record.message for record in caplog.records)

    def test_require_full_coverage_passes_with_complete_coverage(self):
        """Test that require_full_coverage passes when all customers assigned."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
        ]
        cohorts = create_monthly_cohorts(customers)

        assignments = assign_cohorts(customers, cohorts, require_full_coverage=True)

        assert len(assignments) == 2


class TestOverlapValidation:
    """Test validate_non_overlapping() helper function."""

    def test_non_overlapping_cohorts_pass(self):
        """Test that non-overlapping cohorts pass validation."""
        from customer_base_audit.foundation.cohorts import validate_non_overlapping

        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
            CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
            CohortDefinition("2023-03", datetime(2023, 3, 1), datetime(2023, 4, 1)),
        ]

        # Should not raise
        validate_non_overlapping(cohorts)

    def test_overlapping_cohorts_raise_error(self):
        """Test that overlapping cohorts raise ValueError."""
        from customer_base_audit.foundation.cohorts import validate_non_overlapping

        overlapping = [
            CohortDefinition(
                "A", datetime(2023, 1, 1), datetime(2023, 2, 15)
            ),  # Ends mid-Feb
            CohortDefinition(
                "B", datetime(2023, 2, 1), datetime(2023, 3, 1)
            ),  # Starts Feb 1
        ]

        with pytest.raises(ValueError, match="Overlapping cohorts detected"):
            validate_non_overlapping(overlapping)

    def test_overlapping_validation_with_unsorted_input(self):
        """Test that validation works even with unsorted cohorts."""
        from customer_base_audit.foundation.cohorts import validate_non_overlapping

        unsorted_overlapping = [
            CohortDefinition("B", datetime(2023, 2, 1), datetime(2023, 3, 1)),
            CohortDefinition(
                "A", datetime(2023, 1, 1), datetime(2023, 2, 15)
            ),  # Out of order
        ]

        with pytest.raises(ValueError, match="Overlapping cohorts detected"):
            validate_non_overlapping(unsorted_overlapping)

    def test_empty_cohorts_list_passes(self):
        """Test that empty cohorts list passes validation."""
        from customer_base_audit.foundation.cohorts import validate_non_overlapping

        validate_non_overlapping([])  # Should not raise

    def test_single_cohort_passes(self):
        """Test that single cohort passes validation."""
        from customer_base_audit.foundation.cohorts import validate_non_overlapping

        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1))
        ]

        validate_non_overlapping(cohorts)  # Should not raise


class TestCohortCoverageValidation:
    """Test cohort coverage validation utility."""

    def test_validate_full_coverage(self):
        """Test validation when all customers are covered."""
        from customer_base_audit.foundation.cohorts import validate_cohort_coverage

        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
        ]
        cohorts = create_monthly_cohorts(customers)

        assigned, unassigned = validate_cohort_coverage(customers, cohorts)

        assert assigned == 2
        assert unassigned == 0

    def test_validate_partial_coverage(self):
        """Test validation when some customers are outside cohort ranges."""
        from customer_base_audit.foundation.cohorts import validate_cohort_coverage

        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2023, 2, 20), "system"),
            CustomerIdentifier("C3", datetime(2024, 1, 10), "system"),  # Gap!
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
            CohortDefinition("2023-02", datetime(2023, 2, 1), datetime(2023, 3, 1)),
        ]

        assigned, unassigned = validate_cohort_coverage(customers, cohorts)

        assert assigned == 2
        assert unassigned == 1

    def test_validate_no_coverage(self):
        """Test validation when no customers are covered."""
        from customer_base_audit.foundation.cohorts import validate_cohort_coverage

        customers = [
            CustomerIdentifier("C1", datetime(2024, 1, 15), "system"),
            CustomerIdentifier("C2", datetime(2024, 2, 20), "system"),
        ]
        cohorts = [
            CohortDefinition("2023-01", datetime(2023, 1, 1), datetime(2023, 2, 1)),
        ]

        assigned, unassigned = validate_cohort_coverage(customers, cohorts)

        assert assigned == 0
        assert unassigned == 2


class TestPerformance:
    """Test performance characteristics."""

    def test_performance_large_dataset(self):
        """Smoke test: 10k customers, 36 monthly cohorts should complete quickly."""
        import time
        from datetime import timedelta

        # Generate 10k customers spread over 3 years
        customers = [
            CustomerIdentifier(
                f"C{i}", datetime(2020, 1, 1) + timedelta(days=i % 1095), "system"
            )
            for i in range(10000)
        ]

        # Create monthly cohorts
        cohorts = create_monthly_cohorts(customers)
        assert len(cohorts) == 36  # 3 years of monthly cohorts

        # Time the assignment
        start = time.time()
        assignments = assign_cohorts(customers, cohorts)
        elapsed = time.time() - start

        # Verify correctness
        assert len(assignments) == 10000
        # Performance check: should complete in under 1 second on typical hardware
        assert elapsed < 1.0, f"Assignment took {elapsed:.2f}s, expected < 1.0s"
