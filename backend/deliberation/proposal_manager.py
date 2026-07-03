# backend/deliberation/proposal_manager.py

from typing import List, Dict
from backend.state.state_schema import SimulationState


class ProposalManager:
    def __init__(self, agents, memory):
        self.agents = agents
        self.memory = memory

    def generate_proposals(self, state: SimulationState) -> List[Dict]:
        proposals = []

        for agent in self.agents:
            try:
                output = agent.run(state)

                # ✅ enforce structure
                if not isinstance(output, dict):
                    output = {}

                proposal_entry = {
                    "agent": agent.name,
                    "proposal": output,
                    "reasoning": output.get("reasoning", "")
                }

            except Exception as e:
                proposal_entry = {
                    "agent": agent.name,
                    "proposal": {},
                    "reasoning": f"Error: {str(e)}"
                }

            proposals.append(proposal_entry)

            # ✅ log memory
            self.memory.add_entry({
                "type": "proposal",
                "data": proposal_entry
            })

        return proposals