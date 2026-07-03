# backend/hybrid/state_encoder.py
#
# Encodes a SimulationState into the 11-dim observation vector expected by
# the supplied PPO checkpoint. The checkpoint's observation_space is
# Box(low=0, high=1e6, shape=(11,)) which lines up 1:1 with the 11 numeric
# fields already present in SimulationState (state_schema.py):
#
#   [price, quality, budget, reach, conversion_rate,
#    total_customers, active_customers, satisfaction,
#    cash, revenue, expenses]
#
# This ordering is an engineering assumption (no training script was
# supplied with the checkpoint) but it is the natural/only 11-field vector
# this simulator exposes, and the action space (Box(3) in [0,1]) maps
# cleanly onto this simulator's 3 decision variables (price, quality,
# marketing_budget) -- strong evidence the checkpoint was trained on this
# same schema.

from typing import List
import numpy as np

FEATURE_ORDER = [
    "price", "quality", "budget", "reach", "conversion_rate",
    "total_customers", "active_customers", "satisfaction",
    "cash", "revenue", "expenses",
]

OBS_HIGH = 1_000_000.0


def encode_state(state) -> np.ndarray:
    """SimulationState -> np.float32[11], clipped to the checkpoint's obs bounds."""
    values = [
        state.product.price,
        state.product.quality,
        state.marketing.budget,
        state.marketing.reach,
        state.marketing.conversion_rate,
        state.customers.total_customers,
        state.customers.active_customers,
        state.customers.satisfaction,
        state.finance.cash,
        state.finance.revenue,
        state.finance.expenses,
    ]
    arr = np.array(values, dtype=np.float32)
    return np.clip(arr, 0.0, OBS_HIGH)


def feature_order() -> List[str]:
    return list(FEATURE_ORDER)
