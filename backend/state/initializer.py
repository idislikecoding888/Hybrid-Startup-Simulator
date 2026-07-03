# backend/state/initializer.py

from backend.state.state_schema import (
    SimulationState,
    ProductState,
    MarketingState,
    CustomerState,
    FinanceState,
    DecisionState
)
from backend.config import settings


def initialize_state() -> SimulationState:
    return SimulationState(
        step=0,

        product=ProductState(
            price=settings.INITIAL_PRICE,
            quality=0.6  # starting baseline
        ),

        marketing=MarketingState(
            budget=settings.INITIAL_BUDGET * 0.2,  # 20% allocated to marketing
            reach=1000,
            conversion_rate=0.05
        ),

        customers=CustomerState(
            total_customers=settings.INITIAL_CUSTOMERS,
            active_customers=int(settings.INITIAL_CUSTOMERS * 0.8),
            satisfaction=0.6
        ),

        finance=FinanceState(
            cash=settings.INITIAL_BUDGET,
            revenue=0,
            expenses=0
        ),

        decisions=DecisionState(),

        history=[]
    )