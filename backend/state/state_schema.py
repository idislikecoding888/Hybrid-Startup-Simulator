# backend/state/state_schema.py

from pydantic import BaseModel, Field
from typing import List, Dict


class ProductState(BaseModel):
    price: float
    quality: float  # 0 to 1


class MarketingState(BaseModel):
    budget: float
    reach: int
    conversion_rate: float  # 0 to 1


class CustomerState(BaseModel):
    total_customers: int
    active_customers: int
    satisfaction: float  # 0 to 1


class FinanceState(BaseModel):
    cash: float
    revenue: float
    expenses: float


class DecisionState(BaseModel):
    last_founder_decision: Dict = Field(default_factory=dict)
    last_marketing_decision: Dict = Field(default_factory=dict)
    last_investor_decision: Dict = Field(default_factory=dict)


class SimulationState(BaseModel):
    step: int

    product: ProductState
    marketing: MarketingState
    customers: CustomerState
    finance: FinanceState
    decisions: DecisionState
    inventory: dict = Field(default_factory=dict)

    history: List[Dict] = Field(default_factory=list)