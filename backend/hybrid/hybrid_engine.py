from typing import Dict, List

from backend.hybrid.ppo_adapter import PPOAdapter
from backend.hybrid.weighting import compute_agent_weights
from backend.hybrid.fusion import fuse_decision


class HybridDeliberationEngine:
    def __init__(self, agents):
        self.agents = agents
        self.ppo = PPOAdapter()

    def _build_context(self, state) -> Dict:
        latest_history = state.history[-1] if getattr(state, "history", None) else {}
        last_decision = latest_history.get("decision", {}) if isinstance(latest_history, dict) else {}
        last_metrics = latest_history.get("metrics", {}) if isinstance(latest_history, dict) else {}
        last_reward = latest_history.get("reward", None) if isinstance(latest_history, dict) else None

        board_snapshot = (
            f"Step {state.step} | "
            f"price={state.product.price:.2f} | "
            f"quality={state.product.quality:.3f} | "
            f"marketing_budget={state.marketing.budget:.2f} | "
            f"reach={state.marketing.reach} | "
            f"conversion_rate={state.marketing.conversion_rate:.4f} | "
            f"customers={state.customers.total_customers} | "
            f"active_customers={state.customers.active_customers} | "
            f"satisfaction={state.customers.satisfaction:.3f} | "
            f"cash={state.finance.cash:.2f} | "
            f"revenue={state.finance.revenue:.2f} | "
            f"expenses={state.finance.expenses:.2f} | "
            f"inventory_stock={state.inventory.get('stock', 0)}"
        )

        last_step_summary = ""
        if last_decision or last_metrics:
            last_step_summary = (
                f"Last step decision={last_decision} | "
                f"Last metrics={last_metrics} | "
                f"Last reward={last_reward}"
            )

        return {
            "board_snapshot": board_snapshot,
            "last_step_summary": last_step_summary,
            "current_step": state.step,
            "inventory_stock": state.inventory.get("stock", 0),
        }

    def _run_agents(self, state) -> List[Dict]:
        outputs = []
        context = self._build_context(state)

        for agent in self.agents:
            try:
                result = agent.run(state, context=context)
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

        final_decision = fuse_decision(
            agent_outputs=agent_outputs,
            weights=weights,
            state=state,
            reference_decision=reference_decision,
            ppo_value_estimate=ppo_value,
        )

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