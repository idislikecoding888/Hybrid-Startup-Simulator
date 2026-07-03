# backend/metrics/metrics_engine.py

from backend.state.state_schema import SimulationState
from backend.metrics.cac import compute_cac

def compute_metrics(state: SimulationState) -> dict:
    """
    Compute key metrics from current state
    """

    # --- Revenue ---
    revenue = state.finance.revenue

    # --- Estimated new customers ---
    reach = state.marketing.reach
    conversion_rate = state.marketing.conversion_rate

    new_customers = int(reach * conversion_rate)

    # --- CAC ---
    if new_customers > 0:
        

        cac = compute_cac(state.marketing.budget, new_customers)
    else:
        cac = 0

    # --- Conversion ---
    conversion = conversion_rate

    # --- Retention ---
    if state.customers.total_customers > 0:
        retention = state.customers.active_customers / state.customers.total_customers
    else:
        retention = 0

    return {
        "revenue": revenue,
        "cac": cac,
        "conversion_rate": conversion,
        "retention": retention
    }