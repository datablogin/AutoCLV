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

    def test_customers_outside_all_cohorts_not_assigned(self):
        """Test customers with acquisition_ts outside all cohorts are excluded."""
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

        assignments = assign_cohorts(customers, cohorts)

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

    def test_empty_cohorts_list(self):
        """Test assigning with empty cohorts list."""
        customers = [
            CustomerIdentifier("C1", datetime(2023, 1, 15), "system"),
        ]
        cohorts = []

        assignments = assign_cohorts(customers, cohorts)

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
