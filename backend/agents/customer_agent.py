# backend/agents/customer_agent.py

from backend.agents.base.base_agent import BaseAgent
from backend.state.state_schema import SimulationState


class CustomerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="customer",
            goal="Evaluate product value and provide feedback"
        )

    def build_prompt(self, state: SimulationState, context=None) -> str:
        return f"""
You are a group of customers evaluating a product.

Your goal: assess whether the product provides good value.

Current State:
- Price: {state.product.price}
- Quality: {state.product.quality}
- Satisfaction: {state.customers.satisfaction}

Evaluate:
- Is the price justified by quality?
- Are customers satisfied?

Return ONLY JSON in this format:
{{
  "agent": "customer",
  "proposal": {{
    "sentiment": "<positive/neutral/negative>",
    "feedback_score": <0 to 1>,
    "confidence": <number 0 to 1>
  }},
  "reasoning": "<short explanation>"
}}
"""