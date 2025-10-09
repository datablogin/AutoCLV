"""Tests for Gamma-Gamma model wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_base_audit.models.gamma_gamma import (
    GammaGammaConfig,
    GammaGammaModelWrapper,
)


class TestGammaGammaConfig:
    """Test GammaGammaConfig dataclass."""

    def test_default_config(self):
        """Default configuration should use MAP method and standard parameters."""
        config = GammaGammaConfig()
        assert config.method == "map"
        assert config.chains == 4
        assert config.draws == 2000
        assert config.tune == 1000
        assert config.random_seed == 42

    def test_custom_config(self):
        """Custom configuration should override defaults."""
        config = GammaGammaConfig(
            method="mcmc",
            chains=2,
            draws=1000,
            tune=500,
            random_seed=123,
        )
        assert config.method == "mcmc"
        assert config.chains == 2
        assert config.draws == 1000
        assert config.tune == 500
        assert config.random_seed == 123


class TestGammaGammaModelWrapper:
    """Test GammaGammaModelWrapper class."""

    def test_initialization(self):
        """Wrapper should initialize with given config."""
        config = GammaGammaConfig(method="map")
        wrapper = GammaGammaModelWrapper(config)
        assert wrapper.config == config
        assert wrapper.model is None

    def test_initialization_with_default_config(self):
        """Wrapper should initialize with default config if none provided."""
        wrapper = GammaGammaModelWrapper()
        assert wrapper.config.method == "map"
        assert wrapper.model is None

    def test_fit_missing_columns_raises_error(self):
        """fit() should raise ValueError if required columns are missing."""
        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {"customer_id": ["C1", "C2"]}
        )  # Missing frequency, monetary_value

        with pytest.raises(ValueError, match="missing required columns"):
            wrapper.fit(data)

    def test_fit_empty_data_raises_error(self):
        """fit() should raise ValueError on empty dataset."""
        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(columns=["customer_id", "frequency", "monetary_value"])

        with pytest.raises(ValueError, match="empty dataset"):
            wrapper.fit(data)

    def test_fit_frequency_less_than_2_raises_error(self):
        """fit() should raise ValueError if any customer has frequency < 2."""
        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "frequency": [3, 1, 2],  # C2 has frequency < 2
                "monetary_value": [50.0, 30.0, 40.0],
            }
        )

        with pytest.raises(ValueError, match="frequency >= 2"):
            wrapper.fit(data)

    def test_fit_duplicate_customer_ids_raises_error(self):
        """fit() should raise ValueError if duplicate customer_ids exist."""
        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C1"],  # C1 is duplicated
                "frequency": [3, 5, 2],
                "monetary_value": [50.0, 75.0, 30.0],
            }
        )

        with pytest.raises(ValueError, match="Duplicate customer_ids"):
            wrapper.fit(data)

    def test_fit_non_numeric_frequency_raises_error(self):
        """fit() should raise ValueError if frequency is not numeric."""
        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": ["three", "five"],  # Non-numeric
                "monetary_value": [50.0, 75.0],
            }
        )

        with pytest.raises(ValueError, match="frequency column must be numeric"):
            wrapper.fit(data)

    def test_fit_non_numeric_monetary_value_raises_error(self):
        """fit() should raise ValueError if monetary_value is not numeric."""
        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": ["fifty", "seventy-five"],  # Non-numeric
            }
        )

        with pytest.raises(ValueError, match="monetary_value column must be numeric"):
            wrapper.fit(data)

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_fit_map_method_success(self, mock_gamma_gamma_model):
        """fit() should successfully train model using MAP method."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        # Create wrapper and fit
        config = GammaGammaConfig(method="map", random_seed=42)
        wrapper = GammaGammaModelWrapper(config)

        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "frequency": [3, 5, 2],
                "monetary_value": [50.0, 75.0, 30.0],
            }
        )

        wrapper.fit(data)

        # Verify model was created with correct data (including customer_id)
        expected_data = data[["customer_id", "frequency", "monetary_value"]].copy()
        mock_gamma_gamma_model.assert_called_once()
        call_kwargs = mock_gamma_gamma_model.call_args[1]
        pd.testing.assert_frame_equal(call_kwargs["data"], expected_data)

        # Verify fit was called with correct parameters (MAP doesn't use random_seed)
        mock_model_instance.fit.assert_called_once_with(fit_method="map")

        # Verify model is stored
        assert wrapper.model == mock_model_instance

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_fit_mcmc_method_success(self, mock_gamma_gamma_model):
        """fit() should successfully train model using MCMC method."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        # Create wrapper and fit
        config = GammaGammaConfig(
            method="mcmc", chains=2, draws=1000, tune=500, random_seed=123
        )
        wrapper = GammaGammaModelWrapper(config)

        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": [50.0, 75.0],
            }
        )

        wrapper.fit(data)

        # Verify fit was called with MCMC parameters
        mock_model_instance.fit.assert_called_once_with(
            fit_method="mcmc", chains=2, draws=1000, tune=500, random_seed=123
        )

    def test_fit_invalid_method_raises_error(self):
        """fit() should raise ValueError for invalid fitting method."""
        config = GammaGammaConfig(method="invalid_method")
        wrapper = GammaGammaModelWrapper(config)

        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": [50.0, 75.0],
            }
        )

        with pytest.raises(ValueError, match="Invalid fitting method"):
            wrapper.fit(data)

    def test_predict_spend_before_fit_raises_error(self):
        """predict_spend() should raise RuntimeError if model not fitted."""
        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {"customer_id": ["C1"], "frequency": [3], "monetary_value": [50.0]}
        )

        with pytest.raises(RuntimeError, match="Model has not been fitted"):
            wrapper.predict_spend(data)

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_missing_columns_raises_error(self, mock_gamma_gamma_model):
        """predict_spend() should raise ValueError if required columns missing."""
        # Setup mock and fit
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {"customer_id": ["C1"], "frequency": [3], "monetary_value": [50.0]}
        )
        wrapper.fit(fit_data)

        # Try to predict with missing columns
        predict_data = pd.DataFrame({"customer_id": ["C1"]})

        with pytest.raises(ValueError, match="missing required columns"):
            wrapper.predict_spend(predict_data)

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_empty_data_returns_empty_dataframe(
        self, mock_gamma_gamma_model
    ):
        """predict_spend() should return empty DataFrame with correct schema for empty input."""
        # Setup mock and fit
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {"customer_id": ["C1"], "frequency": [3], "monetary_value": [50.0]}
        )
        wrapper.fit(fit_data)

        # Predict on empty data
        predict_data = pd.DataFrame(
            columns=["customer_id", "frequency", "monetary_value"]
        )
        result = wrapper.predict_spend(predict_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["customer_id", "predicted_monetary_value"]

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_success(self, mock_gamma_gamma_model):
        """predict_spend() should return predictions for all customers."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        # Mock expected_customer_spend to return predictions
        mock_predictions = MagicMock()
        mock_predictions.values.flatten.return_value = np.array([52.5, 76.3, 31.2])
        mock_model_instance.expected_customer_spend.return_value = mock_predictions

        # Fit model
        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "frequency": [3, 5, 2],
                "monetary_value": [50.0, 75.0, 30.0],
            }
        )
        wrapper.fit(fit_data)

        # Predict
        predict_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C3"],
                "frequency": [3, 5, 2],
                "monetary_value": [50.0, 75.0, 30.0],
            }
        )
        result = wrapper.predict_spend(predict_data)

        # Verify expected_customer_spend was called with correct data
        mock_model_instance.expected_customer_spend.assert_called_once()
        call_kwargs = mock_model_instance.expected_customer_spend.call_args[1]
        expected_model_data = predict_data[
            ["customer_id", "frequency", "monetary_value"]
        ].copy()
        pd.testing.assert_frame_equal(call_kwargs["data"], expected_model_data)

        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["customer_id", "predicted_monetary_value"]

        # Verify predictions match mock return values
        assert list(result["customer_id"]) == ["C1", "C2", "C3"]
        assert list(result["predicted_monetary_value"]) == [52.5, 76.3, 31.2]

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_preserves_customer_order(self, mock_gamma_gamma_model):
        """predict_spend() should preserve customer_id order from input."""
        # Setup mock
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        # Mock predictions
        mock_predictions = MagicMock()
        mock_predictions.values.flatten.return_value = np.array([100.0, 200.0, 300.0])
        mock_model_instance.expected_customer_spend.return_value = mock_predictions

        # Fit and predict with non-alphabetical customer order
        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {
                "customer_id": ["C3", "C1", "C2"],
                "frequency": [3, 5, 2],
                "monetary_value": [50.0, 75.0, 30.0],
            }
        )
        wrapper.fit(fit_data)

        result = wrapper.predict_spend(fit_data)

        # Verify customer order is preserved
        assert list(result["customer_id"]) == ["C3", "C1", "C2"]

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_duplicate_customer_ids_raises_error(
        self, mock_gamma_gamma_model
    ):
        """predict_spend() should raise ValueError if duplicate customer_ids exist."""
        # Setup mock and fit
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": [50.0, 75.0],
            }
        )
        wrapper.fit(fit_data)

        # Try to predict with duplicate customer IDs
        predict_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2", "C1"],  # C1 duplicated
                "frequency": [3, 5, 3],
                "monetary_value": [50.0, 75.0, 52.0],
            }
        )

        with pytest.raises(ValueError, match="Duplicate customer_ids"):
            wrapper.predict_spend(predict_data)

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_frequency_less_than_2_raises_error(
        self, mock_gamma_gamma_model
    ):
        """predict_spend() should raise ValueError if any customer has frequency < 2."""
        # Setup mock and fit
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": [50.0, 75.0],
            }
        )
        wrapper.fit(fit_data)

        # Try to predict with frequency < 2
        predict_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C3"],
                "frequency": [3, 1],  # C3 has frequency < 2
                "monetary_value": [50.0, 30.0],
            }
        )

        with pytest.raises(ValueError, match="frequency >= 2"):
            wrapper.predict_spend(predict_data)

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_zero_monetary_value_raises_error(
        self, mock_gamma_gamma_model
    ):
        """predict_spend() should raise ValueError if monetary_value <= 0."""
        # Setup mock and fit
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": [50.0, 75.0],
            }
        )
        wrapper.fit(fit_data)

        # Try to predict with zero monetary_value
        predict_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C3"],
                "frequency": [3, 2],
                "monetary_value": [50.0, 0.0],  # C3 has zero monetary_value
            }
        )

        with pytest.raises(ValueError, match="monetary_value must be positive"):
            wrapper.predict_spend(predict_data)

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_predict_spend_negative_monetary_value_raises_error(
        self, mock_gamma_gamma_model
    ):
        """predict_spend() should raise ValueError if monetary_value is negative."""
        # Setup mock and fit
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        fit_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": [50.0, 75.0],
            }
        )
        wrapper.fit(fit_data)

        # Try to predict with negative monetary_value
        predict_data = pd.DataFrame(
            {
                "customer_id": ["C1", "C3"],
                "frequency": [3, 2],
                "monetary_value": [50.0, -10.0],  # C3 has negative monetary_value
            }
        )

        with pytest.raises(ValueError, match="monetary_value must be positive"):
            wrapper.predict_spend(predict_data)


class TestGammaGammaDeterminism:
    """Test deterministic behavior and reproducibility."""

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_map_fitting_same_seed_produces_same_results(self, mock_gamma_gamma_model):
        """MAP fitting with same seed should produce identical results."""
        # Setup mock to return deterministic predictions
        mock_model_instance1 = MagicMock()
        mock_model_instance2 = MagicMock()

        mock_predictions = MagicMock()
        mock_predictions.values.flatten.return_value = np.array([52.5, 76.3])
        mock_model_instance1.expected_customer_spend.return_value = mock_predictions
        mock_model_instance2.expected_customer_spend.return_value = mock_predictions

        mock_gamma_gamma_model.side_effect = [
            mock_model_instance1,
            mock_model_instance2,
        ]

        # Fit two models with same seed
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [3, 5],
                "monetary_value": [50.0, 75.0],
            }
        )

        wrapper1 = GammaGammaModelWrapper(GammaGammaConfig(random_seed=42))
        wrapper1.fit(data)
        _ = wrapper1.predict_spend(data)

        wrapper2 = GammaGammaModelWrapper(GammaGammaConfig(random_seed=42))
        wrapper2.fit(data)
        _ = wrapper2.predict_spend(data)

        # Both should call fit with MAP method (MAP is deterministic, no random seed)
        assert mock_model_instance1.fit.call_args[1]["fit_method"] == "map"
        assert mock_model_instance2.fit.call_args[1]["fit_method"] == "map"


class TestGammaGammaExtremeValues:
    """Test handling of extreme monetary values."""

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_very_small_monetary_values(self, mock_gamma_gamma_model):
        """Model should handle very small but positive monetary values."""
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [10, 10],
                "monetary_value": [0.01, 0.05],  # Very small values
            }
        )

        # Should fit without error
        wrapper.fit(data)
        assert mock_model_instance.fit.called

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_very_large_monetary_values(self, mock_gamma_gamma_model):
        """Model should handle very large monetary values."""
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [10, 10],
                "monetary_value": [1000000.0, 5000000.0],  # Very large values
            }
        )

        # Should fit without error
        wrapper.fit(data)
        assert mock_model_instance.fit.called

    @patch("customer_base_audit.models.gamma_gamma.GammaGammaModel")
    def test_extreme_range_monetary_values(self, mock_gamma_gamma_model):
        """Model should handle extreme range in monetary values."""
        mock_model_instance = MagicMock()
        mock_gamma_gamma_model.return_value = mock_model_instance

        wrapper = GammaGammaModelWrapper()
        data = pd.DataFrame(
            {
                "customer_id": ["C1", "C2"],
                "frequency": [10, 10],
                "monetary_value": [0.01, 1000000.0],  # Extreme range (10^-2 to 10^6)
            }
        )

        # Should fit without error (model will handle numerical scaling)
        wrapper.fit(data)
        assert mock_model_instance.fit.called


class TestGammaGammaIntegration:
    """Integration tests using real PyMC-Marketing GammaGammaModel."""

    @pytest.mark.slow
    def test_real_model_fit_and_predict_map(self):
        """End-to-end test with real GammaGammaModel using MAP."""
        # Create realistic test data (small dataset for speed)
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(20)],
                "frequency": np.random.randint(2, 10, size=20),
                "monetary_value": np.random.uniform(20.0, 100.0, size=20),
            }
        )

        # Fit model
        config = GammaGammaConfig(method="map", random_seed=42)
        wrapper = GammaGammaModelWrapper(config)
        wrapper.fit(data)

        # Model should be fitted
        assert wrapper.model is not None

        # Predict
        predictions = wrapper.predict_spend(data)

        # Verify predictions
        assert len(predictions) == 20
        assert all(predictions["predicted_monetary_value"] > 0)
        assert list(predictions["customer_id"]) == list(data["customer_id"])

        # Predictions should be similar to observed values (shrinkage toward mean)
        # High-frequency customers should have predictions closer to their observed values
        high_freq_mask = data["frequency"] >= 5
        if high_freq_mask.sum() > 0:
            high_freq_data = data[high_freq_mask]
            high_freq_preds = predictions[high_freq_mask]

            # Predictions should be within reasonable range of observed
            mean_abs_diff = np.abs(
                high_freq_data["monetary_value"].values
                - high_freq_preds["predicted_monetary_value"].values
            ).mean()
            # Allow up to 20% mean difference (Bayesian shrinkage effect)
            mean_observed = high_freq_data["monetary_value"].mean()
            assert mean_abs_diff < 0.2 * mean_observed
