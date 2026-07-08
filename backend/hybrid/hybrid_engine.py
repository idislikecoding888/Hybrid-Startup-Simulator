import json
from copy import deepcopy
from typing import Dict, List, Optional

from backend.hybrid.ppo_adapter import PPOAdapter
from backend.hybrid.weighting import compute_agent_weights, rank_agent_proposals
from backend.hybrid.fusion import fuse_decision
from backend.agents.base.memory import AgentMemoryManager


def _clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.5


class HybridDeliberationEngine:
    def __init__(self, agents):
        self.agents = agents
        self.ppo = PPOAdapter()

        # Lightweight online memory for Sprint 2.5.
        self.agent_reputation = {
            "founder": 0.55,
            "marketing": 0.55,
            "investor": 0.55,
            "customer": 0.55,
        }

        self.agent_stats = {
            name: {
                "total_decisions": 0,
                "successful_decisions": 0,
                "average_reward": 0.0,
                "last_reward": 0.0,
                "reputation": 0.55,
            }
            for name in self.agent_reputation
        }

        # Per-agent memory (isolated, latest-8, auto-summarized). This does
        # not add any LLM calls - it only supplies a text block that gets
        # prepended to each agent's existing single prompt/LLM call.
        self.agent_memory = AgentMemoryManager(agent_names=list(self.agent_reputation.keys()))

    def _ensure_agent(self, name: str) -> None:
        if name not in self.agent_reputation:
            self.agent_reputation[name] = 0.55
        if name not in self.agent_stats:
            self.agent_stats[name] = {
                "total_decisions": 0,
                "successful_decisions": 0,
                "average_reward": 0.0,
                "last_reward": 0.0,
                "reputation": self.agent_reputation[name],
            }

    def _update_reputation_from_previous_step(self, state) -> None:
        """
        Uses the most recent history item to slightly update agent reputation.
        This is offline / heuristic memory, not PPO training.
        """
        history = getattr(state, "history", None) or []
        if not history:
            return

        latest_history = history[-1]
        if not isinstance(latest_history, dict):
            return

        reward = latest_history.get("reward")
        deliberation = latest_history.get("deliberation", {}) or {}
        ranking = deliberation.get("proposal_ranking", []) or []
        selected_agent = deliberation.get("selected_agent")

        try:
            reward_val = _clamp01(float(reward))
        except (TypeError, ValueError):
            return

        ranked_map = {}
        for item in ranking:
            if isinstance(item, dict) and item.get("agent"):
                ranked_map[item["agent"]] = item

        step_number = latest_history.get("step")

        for agent_name in list(self.agent_reputation.keys()):
            self._ensure_agent(agent_name)

            rep = float(self.agent_reputation.get(agent_name, 0.55))
            item = ranked_map.get(agent_name, {})
            utility = _clamp01(item.get("utility", 0.5))
            confidence = _clamp01(item.get("confidence", 0.5))
            feasibility = _clamp01(item.get("feasibility", 0.5))

            # Stronger update for the selected agent; weaker for others.
            selected_bonus = 0.10 if agent_name == selected_agent else 0.0

            # Reward signal nudged by how useful the proposal looked.
            signal = reward_val * (0.55 + 0.20 * utility + 0.15 * confidence + 0.10 * feasibility)
            signal = _clamp01(signal + selected_bonus)

            updated_rep = (0.90 * rep) + (0.10 * signal)
            updated_rep = max(0.05, min(0.95, updated_rep))

            self.agent_reputation[agent_name] = updated_rep

            stats = self.agent_stats[agent_name]
            stats["total_decisions"] += 1
            stats["last_reward"] = reward_val
            stats["average_reward"] = (0.85 * stats["average_reward"]) + (0.15 * reward_val)
            stats["reputation"] = updated_rep

            if agent_name == selected_agent and reward_val >= 0.75:
                stats["successful_decisions"] += 1

            # Record this step's outcome into the agent's own isolated
            # memory (latest 8 entries, older ones auto-summarized).
            is_selected = agent_name == selected_agent
            lesson = None
            utility = item.get("utility") if item else None
            if is_selected and reward_val >= 0.6:
                lesson = "Being selected with this proposal led to a strong outcome."
            elif is_selected:
                lesson = "This proposal was selected but the outcome was only modest."
            elif utility is not None and utility < 0.4:
                lesson = "Low utility this round; align proposals more closely with current state."
            elif utility is not None:
                lesson = "Solid proposal, but another agent's plan was favored this round."

            self.agent_memory.add_entry(
                agent_name,
                step=step_number,
                proposal=item.get("proposal", {}) if item else {},
                selected=is_selected,
                reward=reward_val,
                reputation=updated_rep,
                trust_score=updated_rep,
                lesson=lesson,
            )

    def reset_memory(self) -> None:
        """Clears all agents' memory. Called on /simulation/reset."""
        self.agent_memory.reset_all()

    def _format_debate_turn(self, item: Dict) -> str:
        agent = item.get("agent", "unknown")
        proposal = item.get("proposal", {}) or {}
        reasoning = (item.get("reasoning") or "").strip()
        error = item.get("error")

        if agent == "founder":
            summary = (
                f"price={proposal.get('price')}, "
                f"quality={proposal.get('quality')}, "
                f"confidence={proposal.get('confidence')}"
            )
        elif agent == "marketing":
            summary = (
                f"marketing_budget={proposal.get('marketing_budget')}, "
                f"confidence={proposal.get('confidence')}"
            )
        elif agent == "investor":
            summary = (
                f"approve={proposal.get('approve')}, "
                f"risk_level={proposal.get('risk_level')}, "
                f"confidence={proposal.get('confidence')}"
            )
        elif agent == "customer":
            summary = (
                f"sentiment={proposal.get('sentiment')}, "
                f"feedback_score={proposal.get('feedback_score')}, "
                f"confidence={proposal.get('confidence')}"
            )
        else:
            summary = json.dumps(proposal, ensure_ascii=False)

        return (
            f"{agent}: {summary} | "
            f"status={'error' if error else 'ok'} | "
            f"reasoning={reasoning}"
        )


    def _build_debate_synthesis(
        self,
        proposal_ranking: List[Dict],
        reference_decision: Optional[Dict],
        ppo_value: Optional[float],
    ) -> str:
        if not proposal_ranking:
            return "No usable proposals were produced."

        top = proposal_ranking[0]
        runner_up = proposal_ranking[1] if len(proposal_ranking) > 1 else None
        strongest_counter = proposal_ranking[-1]

        lines = [
            (
                f"Selected {top['agent']} because it achieved the strongest combined "
                f"utility ({top.get('utility', 0.0):.3f}), debate score "
                f"({top.get('debate_score', top.get('utility', 0.0)):.3f}), and "
                f"trust ({top.get('reputation', 0.0):.3f})."
            )
        ]

        if runner_up:
            gap = float(top.get("utility", 0.0)) - float(runner_up.get("utility", 0.0))
            lines.append(
                f"The closest alternative was {runner_up['agent']}, showing a real "
                f"trade-off with a utility gap of {gap:.3f}."
            )

        lines.append(
            f"Strongest counterpoint came from {strongest_counter['agent']} with "
            f"utility {strongest_counter.get('utility', 0.0):.3f} and debate score "
            f"{strongest_counter.get('debate_score', strongest_counter.get('utility', 0.0)):.3f}."
        )

        if reference_decision:
            lines.append(
                f"PPO guidance suggested price={reference_decision['price']:.1f}, "
                f"quality={reference_decision['quality']:.3f}, "
                f"marketing_budget={reference_decision['marketing_budget']:.1f}."
            )

        if ppo_value is not None:
            lines.append(f"PPO value estimate={float(ppo_value):.3f}.")

        return " ".join(lines)

    def _build_context(self, state, debate_so_far=None) -> Dict:
        latest_history = state.history[-1] if getattr(state, "history", None) else {}
        last_decision = latest_history.get("decision", {}) if isinstance(latest_history, dict) else {}
        last_metrics = latest_history.get("metrics", {}) if isinstance(latest_history, dict) else {}
        last_reward = latest_history.get("reward", None) if isinstance(latest_history, dict) else None

        live_state = {
            "step": state.step,
            "product": {
                "price": state.product.price,
                "quality": state.product.quality,
            },
            "marketing": {
                "budget": state.marketing.budget,
                "reach": state.marketing.reach,
                "conversion_rate": state.marketing.conversion_rate,
            },
            "customers": {
                "total_customers": state.customers.total_customers,
                "active_customers": state.customers.active_customers,
                "satisfaction": state.customers.satisfaction,
            },
            "finance": {
                "cash": state.finance.cash,
                "revenue": state.finance.revenue,
                "expenses": state.finance.expenses,
            },
            "inventory": {
                "stock": state.inventory.get("stock", 0),
                "production_rate": state.inventory.get("production_rate", 0),
                "capacity": state.inventory.get("capacity", 0),
            },
        }

        board_snapshot = "\n".join([
            f"Step: {live_state['step']}",
            f"Product price: {live_state['product']['price']}",
            f"Product quality: {live_state['product']['quality']}",
            f"Marketing budget: {live_state['marketing']['budget']}",
            f"Reach: {live_state['marketing']['reach']}",
            f"Conversion rate: {live_state['marketing']['conversion_rate']}",
            f"Total customers: {live_state['customers']['total_customers']}",
            f"Active customers: {live_state['customers']['active_customers']}",
            f"Satisfaction: {live_state['customers']['satisfaction']}",
            f"Cash: {live_state['finance']['cash']}",
            f"Revenue: {live_state['finance']['revenue']}",
            f"Expenses: {live_state['finance']['expenses']}",
            f"Inventory stock: {live_state['inventory']['stock']}",
        ])

        last_step_summary = "\n".join([
            f"Last decision: {json.dumps(last_decision, ensure_ascii=False)}",
            f"Last metrics: {json.dumps(last_metrics, ensure_ascii=False)}",
            f"Last reward: {last_reward}",
        ])

        debate_so_far = debate_so_far or []
        debate_lines = [
            "DEBATE CONTEXT (respond directly to prior board arguments):",
            "Say what you agree with, what you disagree with, and why.",
            "Address at least one supporting point and one objection in your reasoning.",
        ]
        for item in debate_so_far[-3:]:
            debate_lines.append(self._format_debate_turn(item))
        debate_summary = "\n".join(debate_lines)

        return {
            "live_state": live_state,
            "live_state_json": json.dumps(live_state, ensure_ascii=False),
            "board_snapshot": board_snapshot,
            "board_snapshot_json": json.dumps(live_state, ensure_ascii=False, indent=2),
            "last_step_summary": last_step_summary,
            "last_step_summary_json": json.dumps({
                "last_decision": last_decision,
                "last_metrics": last_metrics,
                "last_reward": last_reward,
            }, ensure_ascii=False),
            "debate_so_far": debate_so_far,
            "debate_so_far_json": json.dumps(debate_so_far, ensure_ascii=False),
            "debate_so_far_summary": debate_summary,
            "agent_reputation": deepcopy(self.agent_reputation),
            "agent_stats": deepcopy(self.agent_stats),
            "current_step": state.step,
        }
    

    def _run_agents(self, state) -> List[Dict]:
        outputs = []
        debate_so_far = []

        for agent in self.agents:
            context = self._build_context(state, debate_so_far=debate_so_far)
            # Attach this agent's own memory (isolated per agent name) so
            # BaseAgent.run() can prepend it to the existing prompt before
            # the existing single LLM call.
            context["agent_memory_block"] = self.agent_memory.get_prompt_block(agent.name)
            try:
                result = agent.run(state, context=context)
                if not isinstance(result, dict):
                    result = {}

                if "error" in result or "proposal" not in result:
                    output = {
                        "agent": agent.name,
                        "proposal": {},
                        "reasoning": "",
                        "error": result.get("error", "malformed agent output"),
                    }
                else:
                    output = {
                        "agent": agent.name,
                        "proposal": result.get("proposal") or {},
                        "reasoning": result.get("reasoning", ""),
                        "error": None,
                    }
            except Exception as exc:
                output = {
                    "agent": agent.name,
                    "proposal": {},
                    "reasoning": "",
                    "error": str(exc),
                }

            outputs.append(output)
            debate_so_far.append({
                "agent": output["agent"],
                "proposal": output.get("proposal", {}),
                "reasoning": output.get("reasoning", ""),
                "error": output.get("error"),
            })

        return outputs

    def run_deliberation(self, state) -> Dict:
        agent_outputs = self._run_agents(state)

        reference_decision = self.ppo.reference_decision(state)
        ppo_value = self.ppo.value_estimate(state)

        proposal_ranking = rank_agent_proposals(
            agent_outputs,
            reference_decision,
            state,
            agent_reputation=self.agent_reputation,
            agent_stats=self.agent_stats,
        )
        weights = compute_agent_weights(
            agent_outputs,
            reference_decision,
            state,
            agent_reputation=self.agent_reputation,
            agent_stats=self.agent_stats,
        )

        final_decision = fuse_decision(
            agent_outputs=agent_outputs,
            weights=weights,
            state=state,
            reference_decision=reference_decision,
            ppo_value_estimate=ppo_value,
            agent_stats=self.agent_stats,
        )

        ranking_by_agent = {item["agent"]: item for item in proposal_ranking}
        debate_synthesis = self._build_debate_synthesis(
            proposal_ranking=proposal_ranking,
            reference_decision=reference_decision,
            ppo_value=ppo_value,
        )

        reasoning_parts = [
            (
                f"{o['agent']}({weights.get(o['agent'], 0.0):.2f}, "
                f"util={ranking_by_agent.get(o['agent'], {}).get('utility', 0.0):.2f}, "
                f"debate={ranking_by_agent.get(o['agent'], {}).get('debate_score', ranking_by_agent.get(o['agent'], {}).get('utility', 0.0)):.2f}): "
                f"{o.get('reasoning', '')}"
            )
            for o in agent_outputs
            if o["agent"] in weights
        ]

        return {
            "agent_outputs": agent_outputs,
            "ppo_available": self.ppo.available,
            "ppo_reference_decision": reference_decision,
            "ppo_value_estimate": ppo_value,
            "proposal_ranking": proposal_ranking,
            "selected_agent": proposal_ranking[0]["agent"] if proposal_ranking else None,
            "trust_scores": deepcopy(self.agent_reputation),
            "agent_stats": deepcopy(self.agent_stats),
            "weights": weights,
            "final_decision": final_decision,
            "reasoning": " | ".join(reasoning_parts),
            "ppo_learning": self.ppo.learning_summary(),
            "agent_memory": self.agent_memory.diagnostics(),
            "debate_synthesis": debate_synthesis,
            "debate_consensus": proposal_ranking[0].get("consensus_score") if proposal_ranking else None,
        }