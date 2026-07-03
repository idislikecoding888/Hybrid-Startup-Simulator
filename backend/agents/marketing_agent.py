# backend/agents/marketing_agent.py

from backend.agents.base.base_agent import BaseAgent
from backend.state.state_schema import SimulationState


class MarketingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="marketing",
            goal="Maximize customer acquisition efficiently"
        )

    def build_prompt(self, state: SimulationState, context=None) -> str:
        return f"""
You are the Marketing Head of a startup.

Your goal: maximize customer acquisition while keeping costs efficient.

Current State:
- Current Marketing Budget: {state.marketing.budget}
- Reach: {state.marketing.reach}
- Conversion Rate: {state.marketing.conversion_rate}
- Customers: {state.customers.total_customers}
- Cash Available: {state.finance.cash}

Decide:
- marketing_budget (0 to available cash)

Guidelines:
- Higher budget → more reach
- Too high budget → wasteful spending
- Balance growth vs efficiency

Return ONLY JSON in this format:
{{
  "agent": "marketing",
  "proposal": {{
    "marketing_budget": <number>,
    "confidence": <number 0 to 1>
  }},
  "reasoning": "<short explanation>"
}}
"""