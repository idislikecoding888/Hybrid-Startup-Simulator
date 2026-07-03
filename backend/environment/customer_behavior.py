# backend/environment/customer_behavior.py

from backend.state.state_schema import SimulationState


def compute_new_customers(reach: int, conversion_rate: float) -> int:
    """
    Deterministic new customer calculation
    """
    return int(reach * conversion_rate)


def compute_satisfaction(price: float, quality: float) -> float:
    """
    Satisfaction based on value perception
    """

    # Price penalty
    price_penalty = max(0, (price - 500) / 1000)

    # Quality boost
    quality_boost = (quality - 0.5)

    satisfaction = 0.6 + quality_boost - price_penalty

    return max(0, min(1, satisfaction))


def update_customers(state: SimulationState, reach: int, conversion_rate: float):
    """
    Update customer-related fields in state
    """

    # --- New customers ---
    new_customers = compute_new_customers(reach, conversion_rate)

    state.customers.total_customers += new_customers

    # --- Active customers (retention = 80%) ---
    state.customers.active_customers = int(state.customers.total_customers * 0.8)

    # --- Satisfaction ---
    state.customers.satisfaction = compute_satisfaction(
        state.product.price,
        state.product.quality
    )

    return state