"""BG/NBD model wrapper for purchase frequency prediction.

This module wraps PyMC-Marketing's BetaGeoModel (BG/NBD implementation) to predict
customer purchase frequency and probability of being alive. The BG/NBD (Beta-Geometric/
Negative Binomial Distribution) model captures two key customer behaviors:

1. **While-alive transaction process**: Customers make purchases according to a
   Poisson process with rate λ (lambda)
2. **Customer dropout**: After each transaction, customers become inactive with
   probability p, following a geometric distribution

Key assumptions:
- Heterogeneity in transaction rates across customers (Gamma distribution on λ)
- Heterogeneity in dropout probabilities across customers (Beta distribution on p)
- Transaction rate and dropout probability are independent
- Customers can't return after becoming inactive (no reactivation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pymc_marketing.clv import BetaGeoModel


@dataclass
class BGNBDConfig:
    """Configuration for BG/NBD model training.

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
        Random seed for reproducibility (only used if method='mcmc')
    """

    method: str = "map"
    chains: int = 4
    draws: int = 2000
    tune: int = 1000
    random_seed: int = 42


class BGNBDModelWrapper:
    """Wrapper for BG/NBD purchase frequency model.

    The BG/NBD model predicts:
    1. Expected number of future purchases in a given time period
    2. Probability that a customer is still active (not churned)

    These predictions are fundamental inputs to CLV calculation when combined
    with monetary value predictions from the Gamma-Gamma model.

    Examples
    --------
    >>> import pandas as pd
    >>> from customer_base_audit.models.bg_nbd import BGNBDModelWrapper, BGNBDConfig
    >>> # Prepare input data (frequency, recency, T)
    >>> data = pd.DataFrame({
    ...     'customer_id': ['C1', 'C2', 'C3'],
    ...     'frequency': [2, 5, 0],
    ...     'recency': [30.0, 60.0, 0.0],
    ...     'T': [90.0, 90.0, 90.0]
    ... })
    >>> # Train model
    >>> config = BGNBDConfig(method='map')
    >>> wrapper = BGNBDModelWrapper(config)
    >>> wrapper.fit(data)
    >>> # Predict future purchases over next 90 days
    >>> predictions = wrapper.predict_purchases(data, time_periods=90.0)
    >>> predictions.columns
    Index(['customer_id', 'predicted_purchases'], dtype='object')
    >>> # Calculate probability alive
    >>> prob_alive = wrapper.calculate_probability_alive(data)
    >>> prob_alive.columns
    Index(['customer_id', 'prob_alive'], dtype='object')
    """

    def __init__(self, config: BGNBDConfig = BGNBDConfig()) -> None:
        """Initialize BG/NBD model wrapper.

        Parameters
        ----------
        config:
            Model configuration (fitting method, MCMC parameters, etc.)
        """
        self.config = config
        self.model: Optional[BetaGeoModel] = None

    def _validate_bg_nbd_data(
        self, data: pd.DataFrame, operation: str, allow_empty: bool = False
    ) -> None:
        """Validate input data for BG/NBD operations.

        Parameters
        ----------
        data:
            DataFrame to validate
        operation:
            Operation name (for error messages): 'fit', 'predict', 'probability'
        allow_empty:
            If True, allow empty DataFrames (for prediction operations)

        Raises
        ------
        ValueError:
            If data fails validation checks
        """
        # Check required columns
        required_cols = {"customer_id", "frequency", "recency", "T"}
        if not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            raise ValueError(
                f"Input data missing required columns: {missing}. "
                f"Expected columns: {required_cols}"
            )

        # Allow empty for prediction operations
        if data.empty:
            if not allow_empty:
                raise ValueError(
                    f"Cannot {operation} BG/NBD model on empty dataset. "
                    "Provide customer transaction histories."
                )
            return  # Empty is valid for predictions

        # Check for duplicate customer IDs
        if data["customer_id"].duplicated().any():
            duplicates = data[data["customer_id"].duplicated()]["customer_id"].tolist()
            raise ValueError(
                f"Duplicate customer_ids found in {operation} data: {duplicates[:5]}"
                f"{'...' if len(duplicates) > 5 else ''}. "
                "Each customer should appear only once."
            )

        # Validate data types
        if not pd.api.types.is_numeric_dtype(data["frequency"]):
            raise ValueError(
                f"frequency column must be numeric type, got {data['frequency'].dtype}"
            )
        if not pd.api.types.is_numeric_dtype(data["recency"]):
            raise ValueError(
                f"recency column must be numeric type, got {data['recency'].dtype}"
            )
        if not pd.api.types.is_numeric_dtype(data["T"]):
            raise ValueError(f"T column must be numeric type, got {data['T'].dtype}")

        # Check for NaN/Inf values
        for col in ["frequency", "recency", "T"]:
            if data[col].isna().any():
                nan_ids = data[data[col].isna()]["customer_id"].tolist()
                raise ValueError(
                    f"{col} contains NaN values. "
                    f"Found {len(nan_ids)} customers: {nan_ids[:5]}"
                    f"{'...' if len(nan_ids) > 5 else ''}."
                )
            if (data[col] == float("inf")).any() or (data[col] == float("-inf")).any():
                inf_ids = data[
                    (data[col] == float("inf")) | (data[col] == float("-inf"))
                ]["customer_id"].tolist()
                raise ValueError(
                    f"{col} contains infinite values. "
                    f"Found {len(inf_ids)} customers: {inf_ids[:5]}"
                    f"{'...' if len(inf_ids) > 5 else ''}."
                )

        # Validate frequency is integer-valued (counts)
        non_integer_freq = data[data["frequency"] % 1 != 0]
        if not non_integer_freq.empty:
            invalid_ids = non_integer_freq["customer_id"].tolist()
            raise ValueError(
                "frequency must be integer-valued (purchase counts). "
                f"Found {len(invalid_ids)} customers with non-integer frequency: "
                f"{invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}."
            )

        # Validate BG/NBD constraints
        invalid_freq = data[data["frequency"] < 0]
        if not invalid_freq.empty:
            invalid_ids = invalid_freq["customer_id"].tolist()
            raise ValueError(
                "frequency must be non-negative. "
                f"Found {len(invalid_ids)} customers with negative frequency: "
                f"{invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}."
            )

        invalid_recency = data[data["recency"] < 0]
        if not invalid_recency.empty:
            invalid_ids = invalid_recency["customer_id"].tolist()
            raise ValueError(
                "recency must be non-negative. "
                f"Found {len(invalid_ids)} customers with negative recency: "
                f"{invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}."
            )

        invalid_T = data[data["T"] <= 0]
        if not invalid_T.empty:
            invalid_ids = invalid_T["customer_id"].tolist()
            raise ValueError(
                "T must be positive. "
                f"Found {len(invalid_ids)} customers with T <= 0: "
                f"{invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}."
            )

        # Check recency <= T constraint
        invalid_recency_T = data[data["recency"] > data["T"]]
        if not invalid_recency_T.empty:
            invalid_ids = invalid_recency_T["customer_id"].tolist()
            raise ValueError(
                "recency must be <= T (last purchase can't occur after observation end). "
                f"Found {len(invalid_ids)} customers with recency > T: "
                f"{invalid_ids[:5]}{'...' if len(invalid_ids) > 5 else ''}."
            )

    def fit(self, data: pd.DataFrame) -> None:
        """Fit BG/NBD model to customer transaction data.

        The input data should include all customers in the observation period,
        including those with zero repeat purchases (frequency=0).

        Parameters
        ----------
        data:
            DataFrame with columns:
            - customer_id: Unique customer identifier
            - frequency: Number of repeat purchases (total_orders - 1)
            - recency: Time from first purchase to last purchase
            - T: Time from first purchase to observation end

        Raises
        ------
        ValueError:
            If required columns are missing, data types are invalid,
            or duplicate customer IDs exist
        """
        # Validate input using shared validation method
        self._validate_bg_nbd_data(data, operation="fit", allow_empty=False)

        # Create BetaGeoModel instance
        # PyMC-Marketing expects customer_id, frequency, recency, T
        model_data = data[["customer_id", "frequency", "recency", "T"]].copy()

        self.model = BetaGeoModel(data=model_data)

        # Fit model using configured method
        if self.config.method == "map":
            # MAP doesn't use random_seed (deterministic optimization)
            self.model.fit(fit_method="map")
        elif self.config.method == "mcmc":
            self.model.fit(
                fit_method="mcmc",
                chains=self.config.chains,
                draws=self.config.draws,
                tune=self.config.tune,
                random_seed=self.config.random_seed,
            )
            # Check MCMC convergence diagnostics
            self._check_mcmc_convergence()
        else:
            raise ValueError(
                f"Invalid fitting method: {self.config.method}. "
                f"Must be 'map' or 'mcmc'."
            )

    def _check_mcmc_convergence(self) -> None:
        """Check MCMC convergence diagnostics and warn if issues detected.

        Checks R-hat values to ensure chains have converged. Issues a warning
        if any parameter has R-hat > 1.1, which suggests non-convergence.
        """
        import warnings

        import arviz as az

        if not hasattr(self.model, "idata") or self.model.idata is None:
            warnings.warn(
                "MCMC fitting completed but inference data not available. "
                "Cannot check convergence diagnostics."
            )
            return

        try:
            summary = az.summary(self.model.idata)
            if "r_hat" in summary.columns:
                max_rhat = summary["r_hat"].max()
                if max_rhat > 1.1:
                    warnings.warn(
                        f"MCMC chains may not have converged (max R-hat={max_rhat:.3f} > 1.1). "
                        f"Consider increasing tune/draws or checking model specification.",
                        UserWarning,
                    )
        except Exception as e:
            warnings.warn(
                f"Could not check MCMC convergence diagnostics: {e}",
                UserWarning,
            )

    def predict_purchases(
        self, data: pd.DataFrame, time_periods: float
    ) -> pd.DataFrame:
        """Predict expected number of purchases in next time_periods.

        Uses the fitted BG/NBD model to predict future purchase frequency
        for each customer based on their historical RFM metrics.

        Parameters
        ----------
        data:
            DataFrame with columns:
            - customer_id: Unique customer identifier
            - frequency: Number of repeat purchases (must be >= 0)
            - recency: Time from first purchase to last purchase (must be >= 0)
            - T: Time from first purchase to observation end (must be > 0)
        time_periods:
            Prediction horizon (in same units as recency/T, typically days).
            For example, time_periods=365.0 predicts purchases over next year.

        Returns
        -------
        pd.DataFrame:
            DataFrame with columns:
            - customer_id: Customer identifier (preserved from input)
            - predicted_purchases: Expected number of purchases in time_periods

        Raises
        ------
        RuntimeError:
            If model has not been fitted yet (call fit() first)
        ValueError:
            If required columns are missing, constraints are violated,
            duplicate customer IDs exist, or time_periods <= 0
        """
        # Improved model state checking
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call fit() before predict_purchases()."
            )
        if not hasattr(self.model, "idata") or self.model.idata is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before predict_purchases()."
            )

        if time_periods <= 0:
            raise ValueError(
                f"time_periods must be positive, got {time_periods}. "
                "Specify prediction horizon (e.g., 365.0 for one year)."
            )

        # Validate input using shared validation method (allow empty for predictions)
        self._validate_bg_nbd_data(data, operation="predict", allow_empty=True)

        if data.empty:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=["customer_id", "predicted_purchases"])

        # Prepare data for prediction
        model_data = data[["customer_id", "frequency", "recency", "T"]].copy()

        # Predict expected purchases
        # PyMC-Marketing's expected_purchases(t) returns expected purchases over time t
        predictions = self.model.expected_purchases(
            data=model_data, future_t=time_periods
        )

        # Build result DataFrame with customer_id
        result = pd.DataFrame(
            {
                "customer_id": data["customer_id"].values,
                "predicted_purchases": predictions.values.flatten(),
            }
        )

        return result

    def calculate_probability_alive(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate P(customer is still active).

        Uses the fitted BG/NBD model to calculate the probability that each
        customer is still active (has not churned) given their purchase history.

        High-frequency recent buyers have high P(alive), while customers with
        no recent activity have lower P(alive).

        Parameters
        ----------
        data:
            DataFrame with columns:
            - customer_id: Unique customer identifier
            - frequency: Number of repeat purchases (must be >= 0)
            - recency: Time from first purchase to last purchase (must be >= 0)
            - T: Time from first purchase to observation end (must be > 0)

        Returns
        -------
        pd.DataFrame:
            DataFrame with columns:
            - customer_id: Customer identifier (preserved from input)
            - prob_alive: Probability customer is still active (0.0 to 1.0)

        Raises
        ------
        RuntimeError:
            If model has not been fitted yet (call fit() first)
        ValueError:
            If required columns are missing, constraints are violated,
            or duplicate customer IDs exist
        """
        # Improved model state checking
        if self.model is None:
            raise RuntimeError(
                "Model not initialized. Call fit() before calculate_probability_alive()."
            )
        if not hasattr(self.model, "idata") or self.model.idata is None:
            raise RuntimeError(
                "Model has not been fitted. Call fit() before calculate_probability_alive()."
            )

        # Validate input using shared validation method (allow empty for predictions)
        self._validate_bg_nbd_data(data, operation="probability", allow_empty=True)

        if data.empty:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=["customer_id", "prob_alive"])

        # Prepare data for prediction
        model_data = data[["customer_id", "frequency", "recency", "T"]].copy()

        # Calculate probability alive
        prob_alive_values = self.model.expected_probability_alive(data=model_data)

        # Build result DataFrame with customer_id
        result = pd.DataFrame(
            {
                "customer_id": data["customer_id"].values,
                "prob_alive": prob_alive_values.values.flatten(),
            }
        )

        return result
