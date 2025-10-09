"""Tests for BG/NBD model wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from customer_base_audit.models.bg_nbd import BGNBDConfig, BGNBDModelWrapper


class TestBGNBDConfig:
    """Test BGNBDConfig dataclass."""

    def test_default_config(self):
        """Default configuration should use MAP method and standard parameters."""
        config = BGNBDConfig()
        assert config.method == "map"
        assert config.chains == 4
        assert config.draws == 2000
        assert config.tune == 1000
        assert config.random_seed == 42


class TestBGNBDModelWrapper:
    """Test BGNBDModelWrapper class."""

    def test_initialization(self):
        """Wrapper should initialize with given config."""
        config = BGNBDConfig(method="map")
        wrapper = BGNBDModelWrapper(config)
        assert wrapper.config == config
        assert wrapper.model is None

    def test_fit_missing_columns_raises_error(self):
        """fit() should raise ValueError if required columns are missing."""
        wrapper = BGNBDModelWrapper()
        data = pd.DataFrame({"customer_id": ["C1", "C2"]})

        with pytest.raises(ValueError, match="missing required columns"):
            wrapper.fit(data)

    @patch("customer_base_audit.models.bg_nbd.BetaGeoModel")
    def test_fit_map_method_success(self, mock_beta_geo_model):
        """fit() should successfully train model using MAP method."""
        mock_model_instance = MagicMock()
        mock_beta_geo_model.return_value = mock_model_instance

        config = BGNBDConfig(method="map")
        wrapper = BGNBDModelWrapper(config)

        data = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "frequency": [2, 5],
            "recency": [30.0, 60.0],
            "T": [90.0, 90.0],
        })

        wrapper.fit(data)
        mock_model_instance.fit.assert_called_once_with(fit_method="map")

    @patch("customer_base_audit.models.bg_nbd.BetaGeoModel")
    def test_predict_purchases_success(self, mock_beta_geo_model):
        """predict_purchases() should return predictions for all customers."""
        mock_model_instance = MagicMock()
        mock_beta_geo_model.return_value = mock_model_instance

        mock_predictions = MagicMock()
        mock_predictions.values.flatten.return_value = np.array([3.5, 7.2])
        mock_model_instance.expected_purchases.return_value = mock_predictions

        wrapper = BGNBDModelWrapper()
        fit_data = pd.DataFrame({
            "customer_id": ["C1", "C2"],
            "frequency": [2, 5],
            "recency": [30.0, 60.0],
            "T": [90.0, 90.0],
        })
        wrapper.fit(fit_data)

        result = wrapper.predict_purchases(fit_data, time_periods=90.0)

        assert len(result) == 2
        assert list(result.columns) == ["customer_id", "predicted_purchases"]
        assert list(result["predicted_purchases"]) == [3.5, 7.2]

    @pytest.mark.slow
    def test_real_model_fit_and_predict_map(self):
        """End-to-end test with real BetaGeoModel using MAP."""
        np.random.seed(42)
        n_customers = 20
        data = pd.DataFrame({
            "customer_id": [f"C{i}" for i in range(n_customers)],
            "frequency": np.random.randint(0, 10, size=n_customers),
            "recency": np.random.uniform(0.0, 90.0, size=n_customers),
            "T": np.full(n_customers, 90.0),
        })
        data["recency"] = data[["recency", "T"]].min(axis=1)

        config = BGNBDConfig(method="map")
        wrapper = BGNBDModelWrapper(config)
        wrapper.fit(data)

        predictions = wrapper.predict_purchases(data, time_periods=90.0)
        assert len(predictions) == n_customers
        assert all(predictions["predicted_purchases"] >= 0)

        prob_alive = wrapper.calculate_probability_alive(data)
        assert len(prob_alive) == n_customers
        assert all((prob_alive["prob_alive"] >= 0) & (prob_alive["prob_alive"] <= 1))
