"""Pre-configured scenario packs for synthetic data generation.

This module provides ready-to-use scenario configurations for common testing
and demonstration needs. Each scenario represents a realistic business situation
that can be used to validate CLV models and customer-base audit analyses.

Examples
--------
>>> from customer_base_audit.synthetic.scenarios import HIGH_CHURN_SCENARIO
>>> from customer_base_audit.synthetic import generate_customers, generate_transactions
>>> from datetime import date
>>>
>>> customers = generate_customers(1000, date(2023, 1, 1), date(2023, 12, 31))
>>> transactions = generate_transactions(
...     customers,
...     date(2023, 1, 1),
...     date(2023, 12, 31),
...     scenario=HIGH_CHURN_SCENARIO
... )
"""

from datetime import date

from customer_base_audit.synthetic.generator import ScenarioConfig

# Default baseline scenario - moderate behavior, useful for general testing
BASELINE_SCENARIO = ScenarioConfig(
    promo_month=None,
    promo_uplift=1.0,
    launch_date=None,
    churn_hazard=0.08,
    base_orders_per_month=1.2,
    mean_unit_price=30.0,
    price_variability=0.4,
    quantity_mean=1.3,
    seed=None,
)

# High churn scenario - simulates struggling business with customer retention issues
HIGH_CHURN_SCENARIO = ScenarioConfig(
    promo_month=None,
    promo_uplift=1.0,
    launch_date=None,
    churn_hazard=0.30,  # 30% monthly churn - very high
    base_orders_per_month=0.8,  # Lower order frequency
    mean_unit_price=25.0,  # Slightly lower prices
    price_variability=0.5,  # Higher price variance
    quantity_mean=1.1,  # Lower quantities per order
    seed=None,
)

# Product recall scenario - sudden drop in orders during recall month, then recovery
PRODUCT_RECALL_SCENARIO = ScenarioConfig(
    promo_month=6,  # June is the recall month
    promo_uplift=0.3,  # 70% drop in orders (inverse of promotion)
    launch_date=None,
    churn_hazard=0.15,  # Elevated churn during/after recall
    base_orders_per_month=1.5,  # Otherwise healthy order rate
    mean_unit_price=35.0,
    price_variability=0.3,
    quantity_mean=1.4,
    seed=None,
)

# Heavy promotion scenario - extended promotional period with strong uplift
HEAVY_PROMOTION_SCENARIO = ScenarioConfig(
    promo_month=11,  # November (Black Friday / Holiday season)
    promo_uplift=3.0,  # 3x normal order volume
    launch_date=None,
    churn_hazard=0.05,  # Lower churn during promotion
    base_orders_per_month=1.5,
    mean_unit_price=28.0,  # Slightly discounted prices
    price_variability=0.6,  # High variance (deep discounts on some items)
    quantity_mean=2.0,  # Customers buy more items per order
    seed=None,
)

# Product launch scenario - gradual ramp-up after launch date
PRODUCT_LAUNCH_SCENARIO = ScenarioConfig(
    promo_month=None,
    promo_uplift=1.0,
    launch_date=date(2023, 3, 15),  # Mid-March launch
    churn_hazard=0.06,  # Low churn for new exciting product
    base_orders_per_month=0.5,  # Starts slow
    mean_unit_price=45.0,  # Premium product
    price_variability=0.25,  # Consistent pricing
    quantity_mean=1.2,
    seed=None,
)

# Seasonal business scenario - strong promotional spike with moderate churn
SEASONAL_BUSINESS_SCENARIO = ScenarioConfig(
    promo_month=12,  # December peak season
    promo_uplift=2.5,  # 2.5x uplift during peak
    launch_date=None,
    churn_hazard=0.12,  # Moderate churn (seasonal customers)
    base_orders_per_month=0.9,  # Lower baseline (seasonal)
    mean_unit_price=40.0,
    price_variability=0.45,
    quantity_mean=1.5,
    seed=None,
)

# Stable mature business scenario - low churn, consistent behavior
STABLE_BUSINESS_SCENARIO = ScenarioConfig(
    promo_month=None,
    promo_uplift=1.0,
    launch_date=None,
    churn_hazard=0.04,  # Very low churn
    base_orders_per_month=2.0,  # High repeat purchase rate
    mean_unit_price=32.0,
    price_variability=0.3,  # Consistent pricing
    quantity_mean=1.4,
    seed=None,
)

__all__ = [
    "BASELINE_SCENARIO",
    "HIGH_CHURN_SCENARIO",
    "PRODUCT_RECALL_SCENARIO",
    "HEAVY_PROMOTION_SCENARIO",
    "PRODUCT_LAUNCH_SCENARIO",
    "SEASONAL_BUSINESS_SCENARIO",
    "STABLE_BUSINESS_SCENARIO",
]
