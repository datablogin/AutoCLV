"""Tests for CLV Calculator."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_base_audit.models.clv_calculator import CLVCalculator, CLVScore


class TestCLVScore:
    """Test CLVScore dataclass."""

    def test_valid_clv_score(self):
        """CLVScore should accept valid values."""
        score = CLVScore(
            customer_id="C1",
            predicted_purchases=Decimal("5.2"),
            predicted_avg_value=Decimal("50.0"),
            clv=Decimal("78.0"),
            prob_alive=Decimal("0.85"),
        )
        assert score.customer_id == "C1"
        assert score.predicted_purchases == Decimal("5.2")
        assert score.clv == Decimal("78.0")

    def test_negative_predicted_purchases_raises_error(self):
        """CLVScore should reject negative predicted_purchases."""
        with pytest.raises(ValueError, match="predicted_purchases cannot be negative"):
            CLVScore(
                customer_id="C1",
                predicted_purchases=Decimal("-1.0"),
                predicted_avg_value=Decimal("50.0"),
                clv=Decimal("0"),
                prob_alive=Decimal("0.5"),
            )

    def test_negative_clv_raises_error(self):
        """CLVScore should reject negative CLV."""
        with pytest.raises(ValueError, match="clv cannot be negative"):
            CLVScore(
                customer_id="C1",
                predicted_purchases=Decimal("5.0"),
                predicted_avg_value=Decimal("50.0"),
                clv=Decimal("-100.0"),
                prob_alive=Decimal("0.5"),
            )

    def test_prob_alive_out_of_range_raises_error(self):
        """CLVScore should reject prob_alive outside [0, 1]."""
        with pytest.raises(ValueError, match="prob_alive must be between 0 and 1"):
            CLVScore(
                customer_id="C1",
                predicted_purchases=Decimal("5.0"),
                predicted_avg_value=Decimal("50.0"),
                clv=Decimal("100.0"),
                prob_alive=Decimal("1.5"),
            )

    def test_optional_confidence_intervals(self):
        """CLVScore should support optional confidence intervals."""
        score = CLVScore(
            customer_id="C1",
            predicted_purchases=Decimal("5.0"),
            predicted_avg_value=Decimal("50.0"),
            clv=Decimal("100.0"),
            prob_alive=Decimal("0.85"),
            confidence_interval_low=Decimal("80.0"),
            confidence_interval_high=Decimal("120.0"),
        )
        assert score.confidence_interval_low == Decimal("80.0")
        assert score.confidence_interval_high == Decimal("120.0")


class TestCLVCalculator:
    """Test CLVCalculator class."""

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_initialization_with_fitted_models(
        self, mock_gg_wrapper, mock_bg_nbd_wrapper
    ):
        """CLVCalculator should initialize with fitted models."""
        # Mock fitted models
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = MagicMock()
        mock_bg_nbd_instance.model.idata = MagicMock()

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = MagicMock()
        mock_gg_instance.model.idata = MagicMock()

        calculator = CLVCalculator(
            bg_nbd_model=mock_bg_nbd_instance,
            gamma_gamma_model=mock_gg_instance,
            time_horizon_months=12,
            discount_rate=Decimal("0.10"),
            profit_margin=Decimal("0.30"),
        )

        assert calculator.time_horizon_months == 12
        assert calculator.discount_rate == Decimal("0.10")
        assert calculator.profit_margin == Decimal("0.30")
        # Discount factor = 1 / (1.10)^1 ≈ 0.909
        assert abs(calculator.discount_factor - Decimal("0.909")) < Decimal("0.01")

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_initialization_unfitted_bg_nbd_raises_error(
        self, mock_gg_wrapper, mock_bg_nbd_wrapper
    ):
        """CLVCalculator should reject unfitted BG/NBD model."""
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = None

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = MagicMock()
        mock_gg_instance.model.idata = MagicMock()

        with pytest.raises(RuntimeError, match="BG/NBD model has not been fitted"):
            CLVCalculator(
                bg_nbd_model=mock_bg_nbd_instance,
                gamma_gamma_model=mock_gg_instance,
            )

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_initialization_unfitted_gamma_gamma_raises_error(
        self, mock_gg_wrapper, mock_bg_nbd_wrapper
    ):
        """CLVCalculator should reject unfitted Gamma-Gamma model."""
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = MagicMock()
        mock_bg_nbd_instance.model.idata = MagicMock()

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = None

        with pytest.raises(RuntimeError, match="Gamma-Gamma model has not been fitted"):
            CLVCalculator(
                bg_nbd_model=mock_bg_nbd_instance,
                gamma_gamma_model=mock_gg_instance,
            )

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_invalid_time_horizon_raises_error(
        self, mock_gg_wrapper, mock_bg_nbd_wrapper
    ):
        """CLVCalculator should reject non-positive time_horizon_months."""
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = MagicMock()
        mock_bg_nbd_instance.model.idata = MagicMock()

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = MagicMock()
        mock_gg_instance.model.idata = MagicMock()

        with pytest.raises(ValueError, match="time_horizon_months must be positive"):
            CLVCalculator(
                bg_nbd_model=mock_bg_nbd_instance,
                gamma_gamma_model=mock_gg_instance,
                time_horizon_months=0,
            )

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_invalid_profit_margin_raises_error(
        self, mock_gg_wrapper, mock_bg_nbd_wrapper
    ):
        """CLVCalculator should reject profit_margin outside [0, 1]."""
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = MagicMock()
        mock_bg_nbd_instance.model.idata = MagicMock()

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = MagicMock()
        mock_gg_instance.model.idata = MagicMock()

        with pytest.raises(ValueError, match="profit_margin must be between 0 and 1"):
            CLVCalculator(
                bg_nbd_model=mock_bg_nbd_instance,
                gamma_gamma_model=mock_gg_instance,
                profit_margin=Decimal("1.5"),
            )

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_calculate_clv_success(self, mock_gg_wrapper, mock_bg_nbd_wrapper):
        """calculate_clv() should return CLV scores for all customers."""
        # Setup mocked models
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = MagicMock()
        mock_bg_nbd_instance.model.idata = MagicMock()

        # Mock BG/NBD predictions
        mock_purchase_predictions = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "predicted_purchases": [5.0, 8.0, 2.0],
            }
        )
        mock_prob_alive = pd.DataFrame(
            {"customer_id": ["C1", "C2", "C3"], "prob_alive": [0.85, 0.92, 0.45]}
        )
        mock_bg_nbd_instance.predict_purchases.return_value = mock_purchase_predictions
        mock_bg_nbd_instance.calculate_probability_alive.return_value = mock_prob_alive

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = MagicMock()
        mock_gg_instance.model.idata = MagicMock()

        # Mock Gamma-Gamma predictions (only for C1, C2 - C3 is one-time buyer)
        mock_monetary_predictions = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "predicted_monetary_value": [50.0, 75.0],
            }
        )
        mock_gg_instance.predict_spend.return_value = mock_monetary_predictions

        # Create calculator
        calculator = CLVCalculator(
            bg_nbd_model=mock_bg_nbd_instance,
            gamma_gamma_model=mock_gg_instance,
            time_horizon_months=12,
            discount_rate=Decimal("0.10"),
            profit_margin=Decimal("0.30"),
        )

        # Input data
        bg_nbd_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "frequency": [2, 5, 0],
                "recency": [30.0, 60.0, 0.0],
                "T": [90.0, 90.0, 90.0],
            }
        )
        gg_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 6],
                "monetary_value": [50.0, 75.0],
            }
        )

        # Calculate CLV
        result = calculator.calculate_clv(bg_nbd_data, gg_data)

        # Validate result
        assert len(result) == 3
        assert list(result.columns) == [
            "customer_id",
            "predicted_purchases",
            "predicted_avg_value",
            "clv",
            "prob_alive",
        ]

        # Check values
        c1_row = result[result["customer_id"] == "C1"].iloc[0]
        assert c1_row["predicted_purchases"] == 5.0
        assert c1_row["predicted_avg_value"] == 50.0
        assert c1_row["prob_alive"] == 0.85
        # CLV = 5.0 * 50.0 * 0.30 * 0.909 ≈ 68.18
        assert 65.0 < c1_row["clv"] < 70.0

        # C3 (one-time buyer) should have CLV = 0
        c3_row = result[result["customer_id"] == "C3"].iloc[0]
        assert c3_row["predicted_avg_value"] == 0.0
        assert c3_row["clv"] == 0.0

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_calculate_clv_sorted_by_clv_descending(
        self, mock_gg_wrapper, mock_bg_nbd_wrapper
    ):
        """calculate_clv() should sort results by CLV descending."""
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = MagicMock()
        mock_bg_nbd_instance.model.idata = MagicMock()

        mock_purchase_predictions = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "predicted_purchases": [2.0, 8.0, 5.0],
            }
        )
        mock_prob_alive = pd.DataFrame(
            {"customer_id": ["C1", "C2", "C3"], "prob_alive": [0.5, 0.9, 0.8]}
        )
        mock_bg_nbd_instance.predict_purchases.return_value = mock_purchase_predictions
        mock_bg_nbd_instance.calculate_probability_alive.return_value = mock_prob_alive

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = MagicMock()
        mock_gg_instance.model.idata = MagicMock()

        mock_monetary_predictions = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "predicted_monetary_value": [30.0, 100.0, 50.0],
            }
        )
        mock_gg_instance.predict_spend.return_value = mock_monetary_predictions

        calculator = CLVCalculator(
            bg_nbd_model=mock_bg_nbd_instance,
            gamma_gamma_model=mock_gg_instance,
            time_horizon_months=12,
            discount_rate=Decimal("0.10"),
            profit_margin=Decimal("0.30"),
        )

        bg_nbd_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "frequency": [2, 5, 3],
                "recency": [30.0, 60.0, 45.0],
                "T": [90.0, 90.0, 90.0],
            }
        )
        gg_data = bg_nbd_data.copy()
        gg_data["monetary_value"] = [30.0, 100.0, 50.0]

        result = calculator.calculate_clv(bg_nbd_data, gg_data)

        # Result should be sorted by CLV descending
        # C2 has highest (8.0 * 100.0 * 0.30 * discount), followed by C3, then C1
        assert result.iloc[0]["customer_id"] == "C2"
        assert result["clv"].is_monotonic_decreasing

    @patch("customer_base_audit.models.clv_calculator.BGNBDModelWrapper")
    @patch("customer_base_audit.models.clv_calculator.GammaGammaModelWrapper")
    def test_confidence_intervals_not_implemented(
        self, mock_gg_wrapper, mock_bg_nbd_wrapper
    ):
        """calculate_clv() should raise NotImplementedError for confidence intervals."""
        mock_bg_nbd_instance = MagicMock()
        mock_bg_nbd_instance.model = MagicMock()
        mock_bg_nbd_instance.model.idata = MagicMock()

        mock_gg_instance = MagicMock()
        mock_gg_instance.model = MagicMock()
        mock_gg_instance.model.idata = MagicMock()

        calculator = CLVCalculator(
            bg_nbd_model=mock_bg_nbd_instance,
            gamma_gamma_model=mock_gg_instance,
        )

        bg_nbd_data = pd.DataFrame(
            {
                "customer_id": ["C1"],
                "frequency": [2],
                "recency": [30.0],
                "T": [90.0],
            }
        )
        gg_data = pd.DataFrame(
            {"customer_id": ["C1"], "frequency": [3], "monetary_value": [50.0]}
        )

        with pytest.raises(
            NotImplementedError,
            match="Confidence interval calculation not yet implemented",
        ):
            calculator.calculate_clv(
                bg_nbd_data, gg_data, include_confidence_intervals=True
            )

    @pytest.mark.slow
    def test_real_model_end_to_end_clv_calculation(self):
        """End-to-end test with real BG/NBD and Gamma-Gamma models."""
        from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper
        from customer_base_audit.models.gamma_gamma import (
            GammaGammaConfig,
            GammaGammaModelWrapper,
        )

        np.random.seed(42)
        n_customers = 20

        # Create BG/NBD data
        bg_nbd_data = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(n_customers)],
                "frequency": np.random.randint(0, 10, size=n_customers),
                "recency": np.random.uniform(0.0, 90.0, size=n_customers),
                "T": np.full(n_customers, 90.0),
            }
        )
        # Ensure recency <= T
        bg_nbd_data["recency"] = bg_nbd_data[["recency", "T"]].min(axis=1)

        # Create Gamma-Gamma data (only customers with frequency >= 2)
        gg_data = bg_nbd_data[bg_nbd_data["frequency"] >= 1].copy()
        gg_data["frequency"] = gg_data["frequency"] + 1  # Ensure >= 2
        gg_data["monetary_value"] = np.random.uniform(20.0, 100.0, size=len(gg_data))

        # Train models
        bg_nbd_config = BGNBDConfig(method="map")
        bg_nbd_model = BGNBDModelWrapper(bg_nbd_config)
        bg_nbd_model.fit(bg_nbd_data)

        gg_config = GammaGammaConfig(method="map")
        gg_model = GammaGammaModelWrapper(gg_config)
        gg_model.fit(gg_data)

        # Calculate CLV
        calculator = CLVCalculator(
            bg_nbd_model=bg_nbd_model,
            gamma_gamma_model=gg_model,
            time_horizon_months=12,
            discount_rate=Decimal("0.10"),
            profit_margin=Decimal("0.30"),
        )

        result = calculator.calculate_clv(bg_nbd_data, gg_data)

        # Validate results - general checks
        assert len(result) == n_customers
        assert all(result["clv"] >= 0)
        assert all((result["prob_alive"] >= 0) & (result["prob_alive"] <= 1))
        assert all(result["predicted_purchases"] >= 0)
        assert all(result["predicted_avg_value"] >= 0)

        # Result should be sorted by CLV descending
        assert result["clv"].is_monotonic_decreasing

        # Deterministic checks with seed=42
        # These specific values are expected with the fixed seed and model configuration
        assert result["clv"].sum() > 0, "Total CLV should be positive"
        assert result["clv"].max() > result["clv"].min(), "Should have CLV variation"

        # Verify CLV formula: purchases × value × margin × discount
        # Check a sample row to ensure formula is applied correctly
        sample_row = result.iloc[0]
        expected_clv = (
            sample_row["predicted_purchases"]
            * sample_row["predicted_avg_value"]
            * float(calculator.profit_margin)
            * float(calculator.discount_factor)
        )
        # Allow small rounding differences (0.02 tolerance accounts for rounding at 2 decimal places)
        assert abs(sample_row["clv"] - expected_clv) < 0.02, \
            f"CLV formula mismatch: {sample_row['clv']} != {expected_clv}"
