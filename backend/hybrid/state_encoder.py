#
# Encodes a SimulationState into the 11-dim observation vector expected by
# the supplied PPO checkpoint.
#
# Observation order:
#   [price, quality, budget, reach, conversion_rate,
#    total_customers, active_customers, satisfaction,
#    cash, revenue, expenses]
#
# This is kept raw on purpose because the checkpoint was already trained
# against this simulator schema.

from typing import Any, List
import numpy as np

FEATURE_ORDER = [
    "price", "quality", "budget", "reach", "conversion_rate",
    "total_customers", "active_customers", "satisfaction",
    "cash", "revenue", "expenses",
]

OBS_HIGH = 1_000_000.0


def _get_value(obj: Any, path: List[str], default: float = 0.0) -> float:
    current = obj
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)

    try:
        value = float(current)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(value):
        return default

    return value


def encode_state(state) -> np.ndarray:
    """SimulationState -> np.float32[11], clipped to the checkpoint's obs bounds."""
    values = [
        _get_value(state, ["product", "price"]),
        _get_value(state, ["product", "quality"]),
        _get_value(state, ["marketing", "budget"]),
        _get_value(state, ["marketing", "reach"]),
        _get_value(state, ["marketing", "conversion_rate"]),
        _get_value(state, ["customers", "total_customers"]),
        _get_value(state, ["customers", "active_customers"]),
        _get_value(state, ["customers", "satisfaction"]),
        _get_value(state, ["finance", "cash"]),
        _get_value(state, ["finance", "revenue"]),
        _get_value(state, ["finance", "expenses"]),
    ]

    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=OBS_HIGH, neginf=0.0)
    return np.clip(arr, 0.0, OBS_HIGH)


def feature_order() -> List[str]:
    return list(FEATURE_ORDER)