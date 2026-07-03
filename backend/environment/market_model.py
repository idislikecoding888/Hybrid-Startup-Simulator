from backend.state.state_schema import SimulationState


def _pick(value, fallback):
    """
    Safe fallback that does not treat 0 as missing.
    """
    return fallback if value is None else value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def apply_market_dynamics(state: SimulationState, decision: dict) -> SimulationState:
    """
    Applies decision to the environment and returns updated state.
    This version is defensive: it handles missing keys, None values,
    and keeps all decision variables inside valid ranges.
    """
    decision = decision or {}

    # --- Safe extraction with fallback ---
    price = _pick(decision.get("price"), state.product.price if state.product.price is not None else 500)
    quality = _pick(decision.get("quality"), state.product.quality if state.product.quality is not None else 0.5)
    marketing_budget = _pick(
        decision.get("marketing_budget"),
        state.marketing.budget if state.marketing.budget is not None else 1000,
    )

    # --- Enforce safe ranges ---
    price = _clamp(price, 100, 2000)
    quality = _clamp(quality, 0.1, 1.0)
    marketing_budget = _clamp(marketing_budget, 100, 5000)

    # --- Update product ---
    state.product.price = price
    state.product.quality = quality

    # --- Marketing effects ---
    from backend.environment.marketing_model import apply_marketing
    state, reach, conversion_rate = apply_marketing(state, marketing_budget)

    new_customers = int(reach * conversion_rate)

    # --- Customer update ---
    state.customers.total_customers += new_customers
    state.customers.active_customers = int(state.customers.total_customers * 0.8)

    # --- Satisfaction update ---
    price_penalty = max(0, (price - 500) / 1000)
    quality_boost = (quality - 0.5)

    satisfaction = 0.6 + quality_boost - price_penalty
    satisfaction = max(0, min(1, satisfaction))

    state.customers.satisfaction = satisfaction

    # --- Revenue ---
    from backend.environment.inventory_model import update_inventory

    demand = state.customers.active_customers
    actual_sales, state = update_inventory(state, demand)

    revenue = actual_sales * price
    expenses = marketing_budget

    # --- Finance update ---
    state.finance.revenue = revenue
    state.finance.expenses = expenses
    state.finance.cash += (revenue - expenses)

    # --- Marketing state update ---
    state.marketing.budget = marketing_budget
    state.marketing.reach = reach
    state.marketing.conversion_rate = conversion_rate

    return state