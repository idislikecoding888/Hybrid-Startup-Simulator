from typing import List, Dict
from backend.agents.base.base_agent import BaseAgent
from backend.deliberation.conversation_memory import ConversationMemory


class NegotiationManager:
    def __init__(self, agents, memory):
        self.agents = agents
        self.memory = memory

    def negotiate(
        self,
        proposals: List[Dict],
        critiques: List[Dict],
        state: Dict
    ) -> List[Dict]:
        """
        Refines proposals using critiques and produces negotiated proposals.
        """

        negotiated_outputs = []

        for proposal in proposals:
            related_critiques = [
                c for c in critiques
                if c.get("target_agent") == proposal.get("agent")
            ]

            negotiated = self._negotiate_proposal(
                proposal,
                related_critiques,
                state
            )

            negotiated_outputs.append(negotiated)

            # Log into memory
            self.memory.add_entry({
                "type": "negotiation",
                "proposal": proposal,
                "critiques": related_critiques,
                "negotiated_output": negotiated
            })

        return negotiated_outputs

    def _negotiate_proposal(
        self,
        proposal: Dict,
        critiques: List[Dict],
        state: Dict
    ) -> Dict:
        """
        Uses proposer agent to refine its own proposal based on critiques.
        """

        proposer_name = proposal.get("agent")
        agent = self._get_agent(proposer_name)

        prompt = f"""
You are the {proposer_name} agent.

Current State:
{state}

Your original proposal:
{proposal}

Critiques from other agents:
{critiques}

Your task:
- Resolve conflicts raised by critiques
- Adjust your proposal accordingly
- Justify trade-offs (growth vs cost, risk vs reward)

Return STRICT JSON:
{{
  "agent": "{proposer_name}",
  "final_decision": {{
    "price": number,
    "marketing_budget": number,
    "product_quality": number
  }},
  "justification": "text",
  "confidence": 0.0 to 1.0
}}
"""

        response = agent.run(state)

        return self._safe_parse(response, proposer_name)

    def _get_agent(self, name: str) -> BaseAgent:
        for agent in self.agents:
            if agent.name == name:
                return agent
        raise ValueError(f"Agent {name} not found")

    def _safe_parse(self, response: str, agent_name: str) -> Dict:
        """
        Ensures valid structured output even if LLM fails.
        """

        try:
            import json
            return json.loads(response)

        except Exception:
            return {
                "agent": agent_name,
                "proposal": {   # ✅ FIX KEY
                    "price": 500,
                    "marketing_budget": 1000,
                    "product_quality": 0.6
                },
                "justification": "Fallback decision due to parsing error",
                "confidence": 0.5
            }