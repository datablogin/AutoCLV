"""CLV Calculator combining BG/NBD and Gamma-Gamma models.

This module combines purchase frequency predictions (BG/NBD) with monetary value
predictions (Gamma-Gamma) to calculate customer lifetime value (CLV). The calculator
applies profit margins and discount rates to produce actionable CLV scores.

Key formula:
    CLV = (Predicted Purchases) × (Predicted Avg Value) × Profit Margin × Discount Factor

Where:
    - Predicted Purchases: From BG/NBD model (expected future purchases)
    - Predicted Avg Value: From Gamma-Gamma model (expected transaction value)
    - Profit Margin: Percentage of revenue retained as profit (e.g., 0.30 = 30%)
    - Discount Factor: Present value adjustment for time_horizon_months
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import pandas as pd

from customer_base_audit.models.bg_nbd import BGNBDModelWrapper
from customer_base_audit.models.gamma_gamma import GammaGammaModelWrapper


@dataclass(frozen=True)
class CLVScore:
    """CLV prediction for a single customer.

    Attributes
    ----------
    customer_id:
        Unique customer identifier
    predicted_purchases:
        Expected number of future purchases over time horizon (from BG/NBD)
    predicted_avg_value:
        Expected average transaction value (from Gamma-Gamma)
    clv:
        Customer Lifetime Value = purchases × avg_value × profit_margin × discount_factor
    prob_alive:
        Probability customer is still active (from BG/NBD), range [0.0, 1.0]
    confidence_interval_low:
        Lower bound of CLV confidence interval (optional, requires MCMC)
    confidence_interval_high:
        Upper bound of CLV confidence interval (optional, requires MCMC)
    """

    customer_id: str
    predicted_purchases: Decimal
    predicted_avg_value: Decimal
    clv: Decimal
    prob_alive: Decimal
    confidence_interval_low: Optional[Decimal] = None
    confidence_interval_high: Optional[Decimal] = None

    def __post_init__(self) -> None:
        """Validate CLV score values."""
        if self.predicted_purchases < 0:
            raise ValueError(
                f"predicted_purchases cannot be negative: {self.predicted_purchases} "
                f"(customer_id={self.customer_id})"
            )
        if self.predicted_avg_value < 0:
            raise ValueError(
                f"predicted_avg_value cannot be negative: {self.predicted_avg_value} "
                f"(customer_id={self.customer_id})"
            )
        if self.clv < 0:
            raise ValueError(
                f"clv cannot be negative: {self.clv} (customer_id={self.customer_id})"
            )
        if not (0 <= self.prob_alive <= 1):
            raise ValueError(
                f"prob_alive must be between 0 and 1: {self.prob_alive} "
                f"(customer_id={self.customer_id})"
            )
        if (
            self.confidence_interval_low is not None
            and self.confidence_interval_low < 0
        ):
            raise ValueError(
                f"confidence_interval_low cannot be negative: {self.confidence_interval_low} "
                f"(customer_id={self.customer_id})"
            )
        if (
            self.confidence_interval_high is not None
            and self.confidence_interval_high < 0
        ):
            raise ValueError(
                f"confidence_interval_high cannot be negative: {self.confidence_interval_high} "
                f"(customer_id={self.customer_id})"
            )


class CLVCalculator:
    """Calculate CLV by combining BG/NBD and Gamma-Gamma models.

    The CLV calculator integrates purchase frequency and monetary value predictions
    to produce customer lifetime value scores. It handles edge cases like one-time
    buyers and applies business parameters (profit margin, discount rate).

    Examples
    --------
    >>> import pandas as pd
    >>> from customer_base_audit.models.bg_nbd import BGNBDModelWrapper, BGNBDConfig
    >>> from customer_base_audit.models.gamma_gamma import GammaGammaModelWrapper, GammaGammaConfig
    >>> from customer_base_audit.models.clv_calculator import CLVCalculator
    >>> from decimal import Decimal
    >>>
    >>> # Train models (simplified example)
    >>> bg_nbd_data = pd.DataFrame({
    ...     'customer_id': ['C1', 'C2', 'C3'],
    ...     'frequency': [2, 5, 0],
    ...     'recency': [30.0, 60.0, 0.0],
    ...     'T': [90.0, 90.0, 90.0]
    ... })
    >>> gg_data = pd.DataFrame({
    ...     'customer_id': ['C1', 'C2'],
    ...     'frequency': [3, 6],
    ...     'monetary_value': [50.0, 75.0]
    ... })
    >>>
    >>> bg_nbd_model = BGNBDModelWrapper(BGNBDConfig(method='map'))
    >>> bg_nbd_model.fit(bg_nbd_data)
    >>>
    >>> gg_model = GammaGammaModelWrapper(GammaGammaConfig(method='map'))
    >>> gg_model.fit(gg_data)
    >>>
    >>> # Calculate CLV
    >>> calculator = CLVCalculator(
    ...     bg_nbd_model=bg_nbd_model,
    ...     gamma_gamma_model=gg_model,
    ...     time_horizon_months=12,
    ...     discount_rate=Decimal('0.10'),
    ...     profit_margin=Decimal('0.30')
    ... )
    >>> clv_scores = calculator.calculate_clv(bg_nbd_data, gg_data)
    >>> clv_scores.columns
    Index(['customer_id', 'predicted_purchases', 'predicted_avg_value', 'clv', 'prob_alive'], dtype='object')
    """

    def __init__(
        self,
        bg_nbd_model: BGNBDModelWrapper,
        gamma_gamma_model: GammaGammaModelWrapper,
        time_horizon_months: int = 12,
        discount_rate: Decimal = Decimal("0.10"),
        profit_margin: Decimal = Decimal("0.30"),
    ) -> None:
        """Initialize CLV calculator.

        Parameters
        ----------
        bg_nbd_model:
            Fitted BG/NBD model for purchase frequency prediction
        gamma_gamma_model:
            Fitted Gamma-Gamma model for monetary value prediction
        time_horizon_months:
            Prediction horizon in months (default: 12 months = 1 year)
        discount_rate:
            Annual discount rate for present value calculation (default: 0.10 = 10%).
            Applied as: discount_factor = 1 / (1 + rate) ^ (months/12)
        profit_margin:
            Profit margin as decimal (default: 0.30 = 30%).
            Revenue is multiplied by this to get profit contribution.

        Raises
        ------
        ValueError:
            If time_horizon_months <= 0, discount_rate < 0, or profit_margin not in [0, 1]
        RuntimeError:
            If models have not been fitted yet
        """
        # Validate models are fitted
        if bg_nbd_model.model is None or not hasattr(bg_nbd_model.model, "idata"):
            raise RuntimeError(
                "BG/NBD model has not been fitted. Call fit() before creating CLVCalculator."
            )
        if gamma_gamma_model.model is None or not hasattr(
            gamma_gamma_model.model, "idata"
        ):
            raise RuntimeError(
                "Gamma-Gamma model has not been fitted. Call fit() before creating CLVCalculator."
            )

        # Validate parameters
        if time_horizon_months <= 0:
            raise ValueError(
                f"time_horizon_months must be positive, got {time_horizon_months}"
            )
        if discount_rate < 0:
            raise ValueError(f"discount_rate cannot be negative, got {discount_rate}")
        if not (0 <= profit_margin <= 1):
            raise ValueError(
                f"profit_margin must be between 0 and 1, got {profit_margin}"
            )

        self.bg_nbd_model = bg_nbd_model
        self.gamma_gamma_model = gamma_gamma_model
        self.time_horizon_months = time_horizon_months
        self.discount_rate = discount_rate
        self.profit_margin = profit_margin

        # Calculate discount factor once
        # Formula: PV = FV / (1 + r)^t, where t is time horizon in years
        years = Decimal(time_horizon_months) / Decimal(12)
        self.discount_factor = Decimal(1) / ((Decimal(1) + discount_rate) ** years)

    def calculate_clv(
        self,
        bg_nbd_data: pd.DataFrame,
        gamma_gamma_data: pd.DataFrame,
        include_confidence_intervals: bool = False,
    ) -> pd.DataFrame:
        """Calculate CLV for all customers.

        Combines BG/NBD purchase predictions with Gamma-Gamma monetary predictions
        to produce CLV scores. Handles customers with insufficient data for monetary
        predictions (one-time buyers) by setting their CLV to 0.

        **CLV Formula**:
            CLV = (Predicted Purchases) × (Predicted Avg Value) × Profit Margin × Discount Factor

        **Edge Cases**:
        - One-time buyers: No Gamma-Gamma prediction available → CLV = 0
        - Zero predicted purchases: CLV = 0 (customer likely churned)
        - Zero predicted avg value: CLV = 0 (should not occur if Gamma-Gamma fitted correctly)

        Parameters
        ----------
        bg_nbd_data:
            DataFrame with columns [customer_id, frequency, recency, T] for all customers
        gamma_gamma_data:
            DataFrame with columns [customer_id, frequency, monetary_value] for repeat customers
            (excludes one-time buyers)
        include_confidence_intervals:
            If True, calculate confidence intervals (requires MCMC models). Default: False.
            Not yet implemented - reserved for future enhancement.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - customer_id: Customer identifier
            - predicted_purchases: Expected purchases over time horizon
            - predicted_avg_value: Expected average transaction value
            - clv: Customer lifetime value
            - prob_alive: Probability customer is active
            - confidence_interval_low: (optional) Lower bound of CLV CI
            - confidence_interval_high: (optional) Upper bound of CLV CI

            Sorted by CLV descending (highest value customers first).

        Raises
        ------
        ValueError:
            If required columns are missing or data validation fails

        Examples
        --------
        >>> # See class docstring for complete example
        >>> clv_scores = calculator.calculate_clv(bg_nbd_data, gg_data)
        >>> top_customers = clv_scores.head(10)  # Top 10 by CLV
        """
        if include_confidence_intervals:
            raise NotImplementedError(
                "Confidence interval calculation not yet implemented. "
                "Set include_confidence_intervals=False."
            )

        # Convert time horizon to days for BG/NBD prediction
        # Assuming 30.5 days per month on average
        time_periods_days = self.time_horizon_months * 30.5

        # Get BG/NBD predictions (purchases and prob_alive)
        purchase_predictions = self.bg_nbd_model.predict_purchases(
            bg_nbd_data, time_periods=time_periods_days
        )
        prob_alive_predictions = self.bg_nbd_model.calculate_probability_alive(
            bg_nbd_data
        )

        # Get Gamma-Gamma predictions (monetary value)
        monetary_predictions = self.gamma_gamma_model.predict_spend(gamma_gamma_data)

        # Merge predictions
        # Start with all customers from BG/NBD
        result = purchase_predictions.merge(
            prob_alive_predictions, on="customer_id", how="left"
        )

        # Left join with monetary predictions (one-time buyers will have NaN)
        result = result.merge(monetary_predictions, on="customer_id", how="left")

        # Fill NaN monetary values with 0 (one-time buyers get CLV=0)
        result["predicted_monetary_value"] = result["predicted_monetary_value"].fillna(
            0.0
        )

        # Calculate CLV using Decimal for precision
        # CLV = purchases × avg_value × profit_margin × discount_factor
        result["clv"] = result.apply(
            lambda row: (
                Decimal(str(row["predicted_purchases"]))
                * Decimal(str(row["predicted_monetary_value"]))
                * self.profit_margin
                * self.discount_factor
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            axis=1,
        )

        # Rename columns to match CLVScore schema
        result = result.rename(
            columns={
                "predicted_purchases": "predicted_purchases",
                "predicted_monetary_value": "predicted_avg_value",
            }
        )

        # Convert to appropriate types and round
        result["predicted_purchases"] = (
            result["predicted_purchases"].astype(float).round(2)
        )
        result["predicted_avg_value"] = (
            result["predicted_avg_value"].astype(float).round(2)
        )
        result["prob_alive"] = result["prob_alive"].astype(float).round(3)
        result["clv"] = result["clv"].astype(float).round(2)

        # Select final columns
        columns = [
            "customer_id",
            "predicted_purchases",
            "predicted_avg_value",
            "clv",
            "prob_alive",
        ]
        result = result[columns]

        # Sort by CLV descending (highest value customers first)
        result = result.sort_values("clv", ascending=False).reset_index(drop=True)

        return result
