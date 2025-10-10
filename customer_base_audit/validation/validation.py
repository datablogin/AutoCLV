"""CLV model validation framework.

This module provides tools for validating CLV model performance through:
- Temporal train/test splitting for time-series data
- Performance metrics calculation (MAE, MAPE, RMSE, ARPE, R²)
- Time-series cross-validation with expanding windows

Target Performance:
- MAPE < 20% (individual customer level)
- ARPE < 10% (aggregate level)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

# Small epsilon for floating point comparisons to avoid division by zero
_EPSILON = 1e-10


@dataclass(frozen=True)
class ValidationMetrics:
    """Model validation performance metrics.

    Attributes
    ----------
    mae:
        Mean Absolute Error - average absolute difference between actual and predicted
    mape:
        Mean Absolute Percentage Error - average percentage error (%)
    rmse:
        Root Mean Squared Error - square root of average squared errors
    arpe:
        Aggregate Revenue Percent Error - percentage error at aggregate level (%)
    r_squared:
        R² coefficient of determination - proportion of variance explained.
        Can be negative when the model performs worse than predicting the mean.
        Range: (-∞, 1.0], where 1.0 is perfect prediction, 0.0 equals the mean baseline,
        and negative values indicate the model is worse than the mean.
    sample_size:
        Number of samples used in validation

    Notes
    -----
    Target performance metrics:
    - MAPE < 20%: Individual customer predictions are accurate
    - ARPE < 10%: Aggregate revenue predictions are accurate
    - R² > 0.5: Model explains >50% of CLV variance
    - R² < 0: Model performs worse than predicting the mean (needs improvement)
    """

    mae: Decimal
    mape: Decimal
    rmse: Decimal
    arpe: Decimal
    r_squared: Decimal
    sample_size: int

    def __post_init__(self) -> None:
        """Validate metrics are in reasonable ranges.

        Note: R² is not validated because it can be negative when the model
        performs worse than a horizontal line at the mean. A negative R²
        indicates the model is not useful for prediction.
        """
        if self.sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {self.sample_size}")
        if self.mae < 0:
            raise ValueError(f"mae must be non-negative, got {self.mae}")
        if self.mape < 0:
            raise ValueError(f"mape must be non-negative, got {self.mape}")
        if self.rmse < 0:
            raise ValueError(f"rmse must be non-negative, got {self.rmse}")
        if self.arpe < 0:
            raise ValueError(f"arpe must be non-negative, got {self.arpe}")


def temporal_train_test_split(
    transactions: pd.DataFrame,
    train_end_date: datetime,
    observation_end_date: datetime,
    customer_id_col: str = "customer_id",
    date_col: str = "event_ts",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split transaction data temporally for time-series validation.

    This function splits transaction data into:
    - Training set: Transactions up to train_end_date (for model fitting)
    - Observation set: All transactions up to observation_end_date (for RFM calculation)
    - Test set: Transactions between train_end_date and observation_end_date (ground truth)

    Parameters
    ----------
    transactions:
        DataFrame with transaction data including customer_id and timestamp columns
    train_end_date:
        End of training period (exclusive). Model will be fitted on data before this date.
    observation_end_date:
        End of observation period (inclusive). Used to calculate RFM for prediction.
    customer_id_col:
        Name of customer ID column (default: 'customer_id')
    date_col:
        Name of timestamp column (default: 'event_ts')

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_transactions, observation_transactions, test_transactions)
        - train_transactions: For fitting the model
        - observation_transactions: For calculating RFM features
        - test_transactions: For calculating actual future CLV (ground truth)

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> transactions = pd.DataFrame({
    ...     'customer_id': ['C1', 'C1', 'C1', 'C2', 'C2'],
    ...     'event_ts': pd.to_datetime([
    ...         '2023-01-15', '2023-06-20', '2023-12-10',
    ...         '2023-03-10', '2023-11-05'
    ...     ]),
    ...     'amount': [50.0, 75.0, 100.0, 30.0, 45.0]
    ... })
    >>> train, obs, test = temporal_train_test_split(
    ...     transactions,
    ...     train_end_date=datetime(2023, 9, 1),
    ...     observation_end_date=datetime(2023, 12, 31)
    ... )
    >>> len(train)  # Transactions before Sept 1
    3
    >>> len(obs)  # All transactions through Dec 31
    5
    >>> len(test)  # Transactions Sept 1 - Dec 31
    2
    """
    # Validate inputs
    if train_end_date >= observation_end_date:
        raise ValueError(
            f"train_end_date ({train_end_date}) must be before "
            f"observation_end_date ({observation_end_date})"
        )

    required_cols = {customer_id_col, date_col}
    if not required_cols.issubset(transactions.columns):
        missing = required_cols - set(transactions.columns)
        raise ValueError(
            f"transactions missing required columns: {missing}. "
            f"Expected columns: {required_cols}"
        )

    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(transactions[date_col]):
        transactions = transactions.copy()
        transactions[date_col] = pd.to_datetime(transactions[date_col])

    # Split data temporally
    train_mask = transactions[date_col] < train_end_date
    obs_mask = transactions[date_col] <= observation_end_date
    test_mask = (transactions[date_col] >= train_end_date) & (
        transactions[date_col] <= observation_end_date
    )

    train_transactions = transactions[train_mask].copy()
    observation_transactions = transactions[obs_mask].copy()
    test_transactions = transactions[test_mask].copy()

    return train_transactions, observation_transactions, test_transactions


def calculate_clv_metrics(actual: pd.Series, predicted: pd.Series) -> ValidationMetrics:
    """Calculate validation metrics comparing actual vs predicted CLV.

    Computes standard regression metrics for CLV model validation:
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error (excludes zero actual values)
    - RMSE: Root Mean Squared Error
    - ARPE: Aggregate Revenue Percent Error
    - R²: Coefficient of determination

    Parameters
    ----------
    actual:
        Series of actual CLV values (ground truth)
    predicted:
        Series of predicted CLV values from model

    Returns
    -------
    ValidationMetrics
        Dataclass with all computed metrics

    Raises
    ------
    ValueError:
        If actual and predicted have different lengths or contain invalid values

    Notes
    -----
    MAPE Calculation:
        MAPE excludes customers with zero actual CLV to avoid division by zero.
        This means the effective sample size for MAPE may be smaller than the
        reported sample_size. If all actual values are zero, MAPE is set to 0.0.
        This can bias the metric if zeros are systematic (e.g., churned customers).

    Examples
    --------
    >>> import pandas as pd
    >>> actual = pd.Series([100.0, 150.0, 200.0, 50.0])
    >>> predicted = pd.Series([95.0, 160.0, 190.0, 55.0])
    >>> metrics = calculate_clv_metrics(actual, predicted)
    >>> metrics.mae
    Decimal('7.50')
    >>> metrics.mape < Decimal('10.0')  # Less than 10% error
    True
    >>> metrics.r_squared > Decimal('0.9')  # High R²
    True
    """
    # Validate inputs
    if len(actual) != len(predicted):
        raise ValueError(
            f"actual and predicted must have same length: "
            f"{len(actual)} != {len(predicted)}"
        )

    if len(actual) == 0:
        raise ValueError("actual and predicted cannot be empty")

    # Convert to numpy arrays for calculations
    actual_values = actual.values.astype(float)
    predicted_values = predicted.values.astype(float)

    # Check for NaN or inf values
    if np.any(np.isnan(actual_values)) or np.any(np.isinf(actual_values)):
        raise ValueError("actual contains NaN or inf values")
    if np.any(np.isnan(predicted_values)) or np.any(np.isinf(predicted_values)):
        raise ValueError("predicted contains NaN or inf values")

    sample_size = len(actual_values)

    # Calculate MAE: mean(|actual - predicted|)
    absolute_errors = np.abs(actual_values - predicted_values)
    mae = Decimal(str(np.mean(absolute_errors)))

    # Calculate MAPE: mean(|actual - predicted| / |actual|) * 100
    # Handle near-zero actual values by filtering them out for MAPE calculation
    # Use epsilon to avoid both division by exact zero and near-zero values
    nonzero_mask = np.abs(actual_values) > _EPSILON
    if np.sum(nonzero_mask) > 0:
        percentage_errors = (
            absolute_errors[nonzero_mask] / np.abs(actual_values[nonzero_mask])
        ) * 100
        mape = Decimal(str(np.mean(percentage_errors)))
    else:
        # All actual values are (near-)zero - MAPE undefined, use a sentinel value
        mape = Decimal("0.0")

    # Calculate RMSE: sqrt(mean((actual - predicted)^2))
    squared_errors = (actual_values - predicted_values) ** 2
    rmse = Decimal(str(np.sqrt(np.mean(squared_errors))))

    # Calculate ARPE: |sum(actual) - sum(predicted)| / sum(actual) * 100
    total_actual = np.sum(actual_values)
    total_predicted = np.sum(predicted_values)
    if abs(total_actual) > _EPSILON:
        arpe = Decimal(str(abs(total_actual - total_predicted) / total_actual * 100))
    else:
        # Total actual is (near-)zero - ARPE undefined
        arpe = Decimal("0.0")

    # Calculate R²: 1 - (SS_res / SS_tot)
    # SS_res = sum of squared residuals
    # SS_tot = total sum of squares
    ss_res = np.sum(squared_errors)
    mean_actual = np.mean(actual_values)
    ss_tot = np.sum((actual_values - mean_actual) ** 2)

    if abs(ss_tot) > _EPSILON:
        r_squared = Decimal(str(1 - (ss_res / ss_tot)))
    else:
        # All actual values are the same - R² undefined (model explains nothing)
        r_squared = Decimal("0.0")

    return ValidationMetrics(
        mae=mae.quantize(Decimal("0.01")),
        mape=mape.quantize(Decimal("0.01")),
        rmse=rmse.quantize(Decimal("0.01")),
        arpe=arpe.quantize(Decimal("0.01")),
        r_squared=r_squared.quantize(Decimal("0.001")),
        sample_size=sample_size,
    )


def cross_validate_clv(
    transactions: pd.DataFrame,
    model_pipeline: Callable[[pd.DataFrame, datetime], pd.DataFrame],
    n_folds: int = 5,
    time_increment_months: int = 3,
    initial_train_months: int = 12,
    customer_id_col: str = "customer_id",
    date_col: str = "event_ts",
    clv_col: str = "clv",
) -> List[ValidationMetrics]:
    """Perform time-series cross-validation with expanding window.

    This function implements forward-chaining cross-validation for time-series data:
    - Fold 1: Train on months 0-12, test on 13-15
    - Fold 2: Train on months 0-15, test on 16-18
    - Fold 3: Train on months 0-18, test on 19-21
    - etc.

    The training window expands over time (never shrinks), which respects the temporal
    ordering of the data and simulates realistic production scenarios where models are
    retrained on all historical data.

    Parameters
    ----------
    transactions:
        DataFrame with transaction data
    model_pipeline:
        Function that takes (transactions, observation_end_date) and returns
        DataFrame with CLV predictions. Must include customer_id and clv columns.
    n_folds:
        Number of validation folds (default: 5)
    time_increment_months:
        Months between consecutive test periods (default: 3)
    initial_train_months:
        Initial training period in months (default: 12)
    customer_id_col:
        Name of customer ID column (default: 'customer_id')
    date_col:
        Name of timestamp column (default: 'event_ts')
    clv_col:
        Name of CLV column in model output (default: 'clv')

    Returns
    -------
    List[ValidationMetrics]
        List of validation metrics, one per fold

    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> # Create sample transactions over 24 months
    >>> transactions = pd.DataFrame({
    ...     'customer_id': ['C1'] * 20 + ['C2'] * 15,
    ...     'event_ts': pd.date_range('2023-01-01', periods=35, freq='M'),
    ...     'amount': [50.0] * 35
    ... })
    >>> # Define model pipeline
    >>> def my_pipeline(txns, obs_end):
    ...     # Simplified: return constant CLV for demo
    ...     customers = txns['customer_id'].unique()
    ...     return pd.DataFrame({
    ...         'customer_id': customers,
    ...         'clv': [100.0] * len(customers)
    ...     })
    >>> # Run cross-validation
    >>> metrics_list = cross_validate_clv(
    ...     transactions,
    ...     my_pipeline,
    ...     n_folds=3,
    ...     time_increment_months=3,
    ...     initial_train_months=12
    ... )
    >>> len(metrics_list)
    3
    >>> all(isinstance(m, ValidationMetrics) for m in metrics_list)
    True
    """
    # Validate inputs
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if time_increment_months < 1:
        raise ValueError(
            f"time_increment_months must be >= 1, got {time_increment_months}"
        )
    if initial_train_months < 1:
        raise ValueError(
            f"initial_train_months must be >= 1, got {initial_train_months}"
        )

    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(transactions[date_col]):
        transactions = transactions.copy()
        transactions[date_col] = pd.to_datetime(transactions[date_col])

    # Determine date range
    min_date = transactions[date_col].min()
    max_date = transactions[date_col].max()

    # Calculate fold boundaries
    fold_metrics = []

    for fold_idx in range(n_folds):
        # Calculate train end date (expands each fold)
        train_months = initial_train_months + (fold_idx * time_increment_months)
        train_end_date = min_date + pd.DateOffset(months=train_months)

        # Calculate test end date
        test_months = train_months + time_increment_months
        test_end_date = min_date + pd.DateOffset(months=test_months)

        # Skip fold if it extends beyond available data
        if test_end_date > max_date:
            break

        # Split data
        train_txns, obs_txns, test_txns = temporal_train_test_split(
            transactions,
            train_end_date=train_end_date,
            observation_end_date=test_end_date,
            customer_id_col=customer_id_col,
            date_col=date_col,
        )

        # Skip fold if no test data
        if len(test_txns) == 0:
            continue

        # Run model pipeline to get predictions
        # Pipeline receives observation transactions to calculate RFM features
        # The pipeline internally determines which data to use for training
        try:
            predictions = model_pipeline(obs_txns, train_end_date)
        except Exception as e:
            raise RuntimeError(
                f"model_pipeline failed for fold {fold_idx + 1}/{n_folds}: {e}"
            ) from e

        # Validate predictions have required columns
        if customer_id_col not in predictions.columns:
            raise ValueError(
                f"model_pipeline output missing '{customer_id_col}' column"
            )
        if clv_col not in predictions.columns:
            raise ValueError(f"model_pipeline output missing '{clv_col}' column")

        # Calculate actual CLV from test period
        # Test transactions must have 'amount' column for actual revenue calculation
        if "amount" not in test_txns.columns:
            raise ValueError(
                "test_txns must have 'amount' column for actual CLV calculation. "
                f"Found columns: {list(test_txns.columns)}"
            )

        actual_clv = test_txns.groupby(customer_id_col)["amount"].sum()

        # Align predictions with actual CLV (inner join on customer_id)
        comparison = predictions[[customer_id_col, clv_col]].merge(
            actual_clv.reset_index().rename(columns={actual_clv.name: "actual_clv"}),
            on=customer_id_col,
            how="inner",
        )

        # Skip fold if no overlapping customers
        if len(comparison) == 0:
            continue

        # Calculate metrics
        metrics = calculate_clv_metrics(
            actual=comparison["actual_clv"], predicted=comparison[clv_col]
        )

        fold_metrics.append(metrics)

    return fold_metrics
