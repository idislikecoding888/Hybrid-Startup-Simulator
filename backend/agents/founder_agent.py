# backend/agents/founder_agent.py

from backend.agents.base.base_agent import BaseAgent
from backend.state.state_schema import SimulationState


class FounderAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="founder",
            goal="Maximize growth and profitability of the startup"
        )

    def build_prompt(self, state: SimulationState, context=None) -> str:
        return f"""
You are the Founder of a startup.

Your goal: maximize growth and profitability.

Current State:
- Price: {state.product.price}
- Quality: {state.product.quality}
- Customers: {state.customers.total_customers}
- Satisfaction: {state.customers.satisfaction}
- Cash: {state.finance.cash}

Make a decision on:
- price (300–1000)
- quality (0.3–1.0)

IMPORTANT:
- Higher price → more profit but lower demand
- Higher quality → higher satisfaction but costly

Return ONLY JSON in this format:
{{
  "agent": "founder",
  "proposal": {{
    "price": <number>,
    "quality": <number>,
    "confidence": <number 0 to 1>
  }},
  "reasoning": "<short explanation>"
}}
"""