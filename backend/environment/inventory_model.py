# backend/environment/inventory_model.py

from backend.state.state_schema import SimulationState

PRODUCTION_RATE = 200   # units produced per step — enough to cover typical demand
STARTING_STOCK  = 500   # initial stock at simulation start


def initialize_inventory(state: SimulationState) -> SimulationState:
    state.inventory = {
        "stock": STARTING_STOCK,
        "production_rate": PRODUCTION_RATE,
        "capacity": 10_000,
    }
    return state


def produce_inventory(state: SimulationState) -> SimulationState:
    """Add new production each step before sales are applied."""
    if "stock" not in state.inventory:
        state.inventory["stock"] = 0
    if "production_rate" not in state.inventory:
        state.inventory["production_rate"] = PRODUCTION_RATE

    state.inventory["stock"] += state.inventory["production_rate"]
    return state


def apply_sales(state: SimulationState, demand: int):
    """Sell up to available stock; return actual units sold."""
    stock = state.inventory.get("stock", 0)
    actual_sales = min(demand, stock)
    state.inventory["stock"] = stock - actual_sales
    return actual_sales, state


def update_inventory(state: SimulationState, demand: int):
    """
    Full inventory cycle per step:
    1. Produce new stock
    2. Sell against demand
    """
    # Guard: if inventory wasn't initialised (e.g. loaded from old DB row), fix it
    if not state.inventory or "production_rate" not in state.inventory:
        state = initialize_inventory(state)

    state = produce_inventory(state)
    actual_sales, state = apply_sales(state, demand)
    return actual_sales, state
