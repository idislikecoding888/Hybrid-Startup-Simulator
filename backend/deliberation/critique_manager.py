# backend/deliberation/critique_manager.py
# NOTE: This class is retained for reference but is NO LONGER called in the
# active execution path. The single-call DeliberationEngine replaced it.
# Safe to delete once the old architecture is fully retired.

from typing import List, Dict
from backend.agents.base.base_agent import BaseAgent
from backend.agents.base.llm_interface import call_llm
from backend.deliberation.conversation_memory import ConversationMemory
import json


class CritiqueManager:
    def __init__(self, agents, memory):
        self.agents = agents
        self.memory = memory

    def generate_critiques(
        self,
        proposals: List[Dict],
        state: Dict
    ) -> List[Dict]:
        critiques = []

        for proposal in proposals:
            proposer = proposal.get("agent")

            for agent in self.agents:
                if agent.name == proposer:
                    continue

                critique = self._critique_proposal(agent, proposal, state)
                critiques.append(critique)
                self.memory.add_entry({
                    "type": "critique",
                    "critic": agent.name,
                    "target_agent": proposer,
                    "proposal": proposal,
                    "critique": critique,
                })

        return critiques

    def _critique_proposal(
        self,
        agent: BaseAgent,
        proposal: Dict,
        state: Dict
    ) -> Dict:
        prompt = f"""
You are acting as a {agent.name} agent.

Current State:
{state}

Another agent has proposed:
{proposal}

Your task:
- Critically evaluate the proposal
- Identify risks, weaknesses, or unrealistic assumptions
- Suggest improvements if needed

Return STRICT JSON:
{{
  "agent": "{agent.name}",
  "target_agent": "{proposal.get('agent')}",
  "critique": "text",
  "risk_level": "low | medium | high",
  "suggestions": ["point1", "point2"]
}}
"""
        # FIX: use call_llm(prompt) directly — the prompt was previously
        # built but then thrown away by calling agent.run(state) instead.
        response = call_llm(prompt)
        return self._safe_parse(response, agent.name, proposal)

    def _safe_parse(self, response: str, agent_name: str, proposal: Dict) -> Dict:
        try:
            return json.loads(response)
        except Exception:
            return {
                "agent": agent_name,
                "target_agent": proposal.get("agent"),
                "critique": "Parsing failed, default critique generated",
                "risk_level": "medium",
                "suggestions": [],
            }
