# backend/hybrid/hybrid_engine.py
#
# Orchestrates the hybrid LLM + PPO inference flow for a single step:
#
#   1. state is passed in by the caller (already collected)
#   2. run every LLM agent                      -> agent_outputs
#   3/4. encode state for PPO + run PPO inference -> reference_decision
#   5. obtain adaptive weights                  -> weighting.py
#   6. fuse the agent recommendations           -> fusion.py
#   7. produce one final decision
#
# Step 8 (apply decision / update state) stays in engine/step_executor.py,
# unchanged, since this engine's contract (`{"final_decision": {...}}`) is
# identical to the previous single-call DeliberationEngine's, so no other
# file needs to know a hybrid engine is running underneath.
#
# Robustness: a crashing/erroring agent does not abort the step -- its
# weight is redistributed among the surviving agents (weighting.py). If
# PPO is unavailable, weighting.py falls back to equal weights.

from typing import Dict, List

from backend.hybrid.ppo_adapter import PPOAdapter
from backend.hybrid.weighting import compute_agent_weights
from backend.hybrid.fusion import fuse_decision


class HybridDeliberationEngine:
    def __init__(self, agents):
        self.agents = agents  # existing FounderAgent/MarketingAgent/InvestorAgent/CustomerAgent
        self.ppo = PPOAdapter()

    def _run_agents(self, state) -> List[Dict]:
        """
        agent.run() (base_agent.py) returns the agent's parsed JSON as-is
        on success -- i.e. {"agent", "proposal", "reasoning", ["confidence"]}
        matching each agent's own prompt schema -- or
        {"agent", "error", "raw_output"} on a parse failure. We normalize
        both into a common shape here.
        """
        outputs = []
        for agent in self.agents:
            try:
                result = agent.run(state)
                if not isinstance(result, dict):
                    result = {}
                if "error" in result or "proposal" not in result:
                    outputs.append({
                        "agent": agent.name,
                        "proposal": {},
                        "reasoning": "",
                        "error": result.get("error", "malformed agent output"),
                    })
                else:
                    outputs.append({
                        "agent": agent.name,
                        "proposal": result.get("proposal") or {},
                        "reasoning": result.get("reasoning", ""),
                        "error": None,
                    })
            except Exception as e:
                outputs.append({
                    "agent": agent.name,
                    "proposal": {},
                    "reasoning": "",
                    "error": str(e),
                })
        return outputs

    def run_deliberation(self, state) -> Dict:
        agent_outputs = self._run_agents(state)

        reference_decision = self.ppo.reference_decision(state)
        ppo_value = self.ppo.value_estimate(state)

        weights = compute_agent_weights(agent_outputs, reference_decision, state)
        final_decision = fuse_decision(agent_outputs, weights, state)

        reasoning_parts = [
            f"{o['agent']}({weights.get(o['agent'], 0.0):.2f}): {o.get('reasoning', '')}"
            for o in agent_outputs
            if o["agent"] in weights
        ]

        return {
            "agent_outputs": agent_outputs,
            "ppo_available": self.ppo.available,
            "ppo_reference_decision": reference_decision,
            "ppo_value_estimate": ppo_value,
            "weights": weights,
            "final_decision": final_decision,
            "reasoning": " | ".join(reasoning_parts),
        }
