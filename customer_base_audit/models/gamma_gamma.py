"""Gamma-Gamma model wrapper for monetary value prediction.

This module wraps PyMC-Marketing's GammaGammaModel to predict customer-level
average transaction values. The Gamma-Gamma model assumes each customer has
a latent mean transaction value, with individual transactions varying randomly
around that mean.

Key assumptions:
- Transaction values are independent of purchase frequency
- Customer monetary values follow a Gamma distribution
- Within-customer transaction values vary around their mean
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pymc_marketing.clv import GammaGammaModel


@dataclass
class GammaGammaConfig:
    """Configuration for Gamma-Gamma model training.

    Attributes
    ----------
    method:
        Fitting method. 'map' for fast Maximum A Posteriori estimation,
        'mcmc' for full Bayesian inference with sampling.
    chains:
        Number of MCMC chains (only used if method='mcmc')
    draws:
        Number of samples per chain (only used if method='mcmc')
    tune:
        Number of tuning samples per chain (only used if method='mcmc')
    random_seed:
        Random seed for reproducibility
    """

    method: str = "map"
    chains: int = 4
    draws: int = 2000
    tune: int = 1000
    random_seed: int = 42


class GammaGammaModelWrapper:
    """Wrapper for Gamma-Gamma monetary value model.

    The Gamma-Gamma model predicts expected average transaction value for each
    customer based on their historical spending patterns. It requires customers
    with at least 2 transactions to estimate a meaningful average.

    Examples
    --------
    >>> import pandas as pd
    >>> from customer_base_audit.models.gamma_gamma import GammaGammaModelWrapper, GammaGammaConfig
    >>> # Prepare input data (requires frequency >= 2)
    >>> data = pd.DataFrame({
    ...     'customer_id': ['C1', 'C2', 'C3'],
    ...     'frequency': [3, 5, 2],
    ...     'monetary_value': [50.0, 75.0, 30.0]
    ... })
    >>> # Train model
    >>> config = GammaGammaConfig(method='map')
    >>> wrapper = GammaGammaModelWrapper(config)
    >>> wrapper.fit(data)
    >>> # Predict average transaction values
    >>> predictions = wrapper.predict_spend(data)
    >>> predictions.columns
    Index(['customer_id', 'predicted_monetary_value'], dtype='object')
    """

    def __init__(self, config: GammaGammaConfig = GammaGammaConfig()) -> None:
        """Initialize Gamma-Gamma model wrapper.

        Parameters
        ----------
        config:
            Model configuration (fitting method, MCMC parameters, etc.)
        """
        self.config = config
        self.model: Optional[GammaGammaModel] = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit Gamma-Gamma model to customer spending data.

        The input data must contain customers with frequency >= 2. One-time buyers
        should be excluded upstream (see prepare_gamma_gamma_inputs).

        Parameters
        ----------
        data:
            DataFrame with columns:
            - customer_id: Unique customer identifier
            - frequency: Number of transactions (must be >= 2)
            - monetary_value: Average transaction value

        Raises
        ------
        ValueError:
            If required columns are missing or if any customer has frequency < 2
        """
        # Validate input
        required_cols = {"customer_id", "frequency", "monetary_value"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(
                f"Input data missing required columns: {missing}. "
                f"Expected columns: {required_cols}"
            )

        if data.empty:
            raise ValueError(
                "Cannot fit Gamma-Gamma model on empty dataset. "
                "Ensure at least some customers have frequency >= 2."
            )

        # Validate frequency >= 2 (Gamma-Gamma requirement)
        invalid_freq = data[data["frequency"] < 2]
        if not invalid_freq.empty:
            invalid_ids = invalid_freq["customer_id"].tolist()
            raise ValueError(
                f"Gamma-Gamma model requires frequency >= 2. "
                f"Found {len(invalid_ids)} customers with frequency < 2: "
                f"{invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}. "
                f"Use prepare_gamma_gamma_inputs with min_frequency=2 to filter."
            )

        # Create GammaGammaModel instance
        # PyMC-Marketing expects customer_id, frequency, and monetary_value
        model_data = data[["customer_id", "frequency", "monetary_value"]].copy()

        self.model = GammaGammaModel(data=model_data)

        # Fit model using configured method
        if self.config.method == "map":
            self.model.fit(fit_method="map", random_seed=self.config.random_seed)
        elif self.config.method == "mcmc":
            self.model.fit(
                fit_method="mcmc",
                chains=self.config.chains,
                draws=self.config.draws,
                tune=self.config.tune,
                random_seed=self.config.random_seed,
            )
        else:
            raise ValueError(
                f"Invalid fitting method: {self.config.method}. "
                f"Must be 'map' or 'mcmc'."
            )

    def predict_spend(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict expected average transaction value per customer.

        Uses the fitted Gamma-Gamma model to predict each customer's latent
        mean transaction value based on their observed spending history.

        Parameters
        ----------
        data:
            DataFrame with columns:
            - customer_id: Unique customer identifier
            - frequency: Number of transactions
            - monetary_value: Observed average transaction value

        Returns
        -------
        pd.DataFrame:
            DataFrame with columns:
            - customer_id: Customer identifier (preserved from input)
            - predicted_monetary_value: Predicted average transaction value

        Raises
        ------
        RuntimeError:
            If model has not been fitted yet (call fit() first)
        ValueError:
            If required columns are missing
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_spend()."
            )

        # Validate input
        required_cols = {"customer_id", "frequency", "monetary_value"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(
                f"Input data missing required columns: {missing}. "
                f"Expected columns: {required_cols}"
            )

        if data.empty:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                columns=["customer_id", "predicted_monetary_value"]
            )

        # Prepare data for prediction (PyMC-Marketing expects customer_id, frequency, monetary_value)
        model_data = data[["customer_id", "frequency", "monetary_value"]].copy()

        # Get expected conditional average
        # This is the customer's predicted mean transaction value
        predictions = self.model.expected_customer_spend(
            customer_data=model_data
        )

        # Build result DataFrame with customer_id
        result = pd.DataFrame(
            {
                "customer_id": data["customer_id"].values,
                "predicted_monetary_value": predictions.values.flatten(),
            }
        )

        return result
