from backend.agents.base.base_agent import BaseAgent
from backend.state.state_schema import SimulationState


class CustomerAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="customer",
            goal="Evaluate product value and provide feedback"
        )

    def build_prompt(self, state: SimulationState, context=None) -> str:
        live_state_json = ""
        last_step_summary_json = ""
        debate_so_far = "[]"

        if context and isinstance(context, dict):
            live_state_json = context.get("live_state_json", "")
            last_step_summary_json = context.get("last_step_summary_json", "")
            debate_so_far = context.get("debate_so_far", []) or []
            debate_so_far = __import__("json").dumps(debate_so_far, ensure_ascii=False)

        return f"""
You are a customer representative speaking for real buyers in a live startup board meeting.

Be blunt, practical, and human. If the price feels too high, say it. If the quality feels worth it, say it.
Do not sound neutral or robotic. Respond like an actual buyer reacting to value, trust, fairness, and product quality.

GROUNDED INPUTS:
LIVE_STATE_JSON = {live_state_json}
LAST_STEP_SUMMARY_JSON = {last_step_summary_json}
DEBATE_SO_FAR_JSON = {debate_so_far}

Rules:
- Use only the values in LIVE_STATE_JSON and LAST_STEP_SUMMARY_JSON.
- Do NOT reuse older launch-state numbers.
- Do NOT invent satisfaction or price values.
- If you disagree with the founder or marketing lead, say so in human language.
- Mention at least two exact current numbers from LIVE_STATE_JSON.
- If the price looks unfair, say that plainly.
- If the quality genuinely feels strong, say that plainly.

Evaluate:
- Is the price justified by quality?
- Are customers satisfied?
- Would a real buyer keep buying or walk away?

Return ONLY JSON in this format:
{{
  "agent": "customer",
  "proposal": {{
    "sentiment": "<positive/neutral/negative>",
    "feedback_score": <0 to 1>,
    "confidence": <number 0 to 1>
  }},
  "reasoning": "<short human-like explanation>"
}}
"""
