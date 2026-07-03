from typing import List, Dict


class ConsensusManager:
    def __init__(self, memory):
        self.memory = memory

    def reach_consensus(self, negotiated_proposals: List[Dict]) -> Dict:
        """
        Combine all proposals into a final decision.
        Deterministic + safe merge logic.
        """

        final_decision = {
            "price": 500,              # safe defaults
            "quality": 0.5,
            "marketing_budget": 1000,
            "decision_log": []
        }

        for proposal in negotiated_proposals:
            agent = proposal.get("agent")
            data = proposal.get("proposal", {})

            if not isinstance(data, dict):
                continue

            # Founder controls product
            if agent == "founder":
                p = data.get("price")
                q = data.get("quality")

                if p is not None:
                    final_decision["price"] = p

                if q is not None:
                    final_decision["quality"] = q

            # Marketing controls budget
            elif agent == "marketing":
                mb = data.get("marketing_budget")
                if mb is not None:
                    final_decision["marketing_budget"] = mb

            # Investor veto logic
            elif agent == "investor":
                if data.get("reject") is True:
                    final_decision["decision_log"].append("Investor rejected proposal")

            # Customer signal
            elif agent == "customer":
                final_decision["decision_log"].append("Customer feedback considered")

        return final_decision