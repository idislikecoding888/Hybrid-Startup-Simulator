# backend/deliberation/deliberation_engine.py

from typing import Dict
from backend.agents.base.llm_interface import call_llm
import json
import re


class DeliberationEngine:
    def __init__(self, agents):
        self.agents = agents

    def run_deliberation(self, state) -> Dict:
        """
        Single LLM call that simulates a full round-table discussion
        between 4 agents and returns a concrete business decision.
        """

        # Build a readable state summary instead of dumping the raw Pydantic object
        state_summary = f"""
- Step: {state.step}
- Product price: ${state.product.price}  |  Quality score: {state.product.quality:.2f}
- Marketing budget: ${state.marketing.budget}  |  Reach: {state.marketing.reach}  |  Conversion: {state.marketing.conversion_rate:.1%}
- Customers: {state.customers.total_customers} total, {state.customers.active_customers} active, satisfaction {state.customers.satisfaction:.2f}
- Financials: cash ${state.finance.cash:.0f}  |  revenue ${state.finance.revenue:.0f}  |  expenses ${state.finance.expenses:.0f}
- Inventory stock: {state.inventory.get('stock', 0)}
"""

        prompt = f"""You are simulating a startup leadership team making a real business decision.

CURRENT COMPANY STATE:
{state_summary}

THE ROUND TABLE — 4 agents debate, push back, and reach a decision:

1. FOUNDER — cares about product quality, long-term brand, sustainable pricing
2. MARKETING — cares about customer acquisition, reach, conversion rate, CAC
3. INVESTOR — cares about ROI, burn rate, profitability, risk
4. CUSTOMER REP — speaks for buyers: price sensitivity, satisfaction, value perception

INSTRUCTIONS:
- Each agent speaks 1-2 times in sequence
- Agents must DISAGREE with each other where their interests conflict
- Reference the actual numbers from the state above (e.g. "our cash is down to X", "satisfaction is only Y")
- After debate, arrive at a FINAL DECISION with specific numbers
- Price must be between 100 and 2000
- Quality must be between 0.1 and 1.0
- Marketing budget must be between 100 and 5000

Return ONLY this JSON, no other text:

{{
  "discussion": [
    "Founder: <specific point referencing current state>",
    "Marketing: <push back or agree with reasoning>",
    "Investor: <financial concern or approval>",
    "Customer Rep: <buyer perspective>",
    "Founder: <response or concession>",
    "Investor: <final stance>"
  ],
  "final_decision": {{
    "price": <number>,
    "quality": <number between 0.1 and 1.0>,
    "marketing_budget": <number>
  }},
  "reasoning": "<one sentence summary of the decision logic>"
}}"""

        response = call_llm(prompt)

        try:
            # Strip markdown code fences if LLM wraps in ```json ... ```
            cleaned = re.sub(r"```(?:json)?|```", "", response).strip()
            parsed = json.loads(cleaned)

            if "final_decision" not in parsed:
                raise ValueError("Missing final_decision")

            # Validate and clamp decision values to safe ranges
            fd = parsed["final_decision"]
            fd["price"] = max(100, min(2000, float(fd.get("price", 500))))
            fd["quality"] = max(0.1, min(1.0, float(fd.get("quality", 0.5))))
            fd["marketing_budget"] = max(100, min(5000, float(fd.get("marketing_budget", 1000))))
            parsed["final_decision"] = fd

            return parsed

        except Exception as e:
            # Surface the actual error so it's not invisible
            return {
                "discussion": [f"[System: LLM parse failed — {str(e)}. Raw response: {response[:200]}]"],
                "final_decision": {
                    "price": state.product.price,
                    "quality": state.product.quality,
                    "marketing_budget": state.marketing.budget,
                },
                "reasoning": "Fallback: held previous state due to LLM error",
            }
