from backend.state.state_schema import SimulationState


def apply_market_dynamics(state: SimulationState, decision: dict) -> SimulationState:
    """
    Applies decision to the environment and returns updated state
    """

    # --- Safe extraction with fallback ---
    price = decision.get("price") or state.product.price or 500
    quality = decision.get("quality") or state.product.quality or 0.5
    marketing_budget = decision.get("marketing_budget") or state.marketing.budget or 1000

    # --- HARD GUARD (critical) ---
    if quality is None:
        quality = 0.5
    if price is None:
        price = 500
    if marketing_budget is None:
        marketing_budget = 1000

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