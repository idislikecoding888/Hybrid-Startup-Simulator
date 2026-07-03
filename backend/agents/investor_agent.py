# backend/agents/investor_agent.py

from backend.agents.base.base_agent import BaseAgent
from backend.state.state_schema import SimulationState


class InvestorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="investor",
            goal="Ensure sustainable growth and protect capital"
        )

    def build_prompt(self, state: SimulationState, context=None) -> str:
        return f"""
You are an Investor evaluating a startup.

Your goal: ensure sustainable growth and protect capital.

Current State:
- Cash: {state.finance.cash}
- Revenue: {state.finance.revenue}
- Expenses: {state.finance.expenses}
- Customers: {state.customers.total_customers}
- Satisfaction: {state.customers.satisfaction}

Evaluate:
- Is the startup spending efficiently?
- Is growth sustainable?
- Is cash at risk?

Return ONLY JSON in this format:
{{
  "agent": "investor",
  "proposal": {{
    "approve": <true/false>,
    "risk_level": "<low/medium/high>",
    "confidence": <number 0 to 1>
  }},
  "reasoning": "<short explanation>"
}}
"""