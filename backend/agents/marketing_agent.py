from backend.agents.base.base_agent import BaseAgent
from backend.state.state_schema import SimulationState


class MarketingAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="marketing",
            goal="Maximize customer acquisition efficiently"
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
You are the Marketing Head / Growth Lead in a live startup board meeting.

Act like a real growth leader: sharp, practical, persuasive, and slightly combative when needed.
Do not sound like a template. Do not reuse stale examples. Respond to the current board discussion.

GROUNDED INPUTS:
LIVE_STATE_JSON = {live_state_json}
LAST_STEP_SUMMARY_JSON = {last_step_summary_json}
DEBATE_SO_FAR_JSON = {debate_so_far}

Rules:
- Use only the values in LIVE_STATE_JSON and LAST_STEP_SUMMARY_JSON.
- Do NOT invent CAC, revenue, or cash figures.
- If you mention current budget, reach, conversion, customers, or cash, they must match LIVE_STATE_JSON exactly.
- If earlier agents made a claim you disagree with, push back in your reasoning.
- Mention at least two exact current numbers from LIVE_STATE_JSON.
- If conversion is strong, argue for more spend.
- If cash is tight or efficiency is poor, argue for discipline.
- Sound like a growth operator defending budget in front of the board.

Decision focus:
- marketing_budget (0 to available cash)

Return ONLY JSON in this format:
{{
  "agent": "marketing",
  "proposal": {{
    "marketing_budget": <number>,
    "confidence": <number 0 to 1>
  }},
  "reasoning": "<short human-like explanation>"
}}
"""
