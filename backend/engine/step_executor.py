from backend.environment.market_model import apply_market_dynamics
from backend.metrics.metrics_engine import compute_metrics
from backend.hybrid.reward import compute_reward
from backend.storage.logs_repo import LogsRepository


class StepExecutor:
    def __init__(self, deliberation_service):
        self.deliberation_service = deliberation_service
        self._logs_repo = LogsRepository()

    def _store_agent_decisions(self, state, deliberation_output):
        """
        Persist the latest agent proposals into state.decisions so the UI
        and history can inspect them later.
        """
        if not hasattr(state, "decisions") or state.decisions is None:
            return

        agent_map = {}
        for item in deliberation_output.get("agent_outputs", []) or []:
            if not isinstance(item, dict):
                continue
            agent_name = item.get("agent")
            if not agent_name:
                continue
            agent_map[agent_name] = {
                "proposal": item.get("proposal", {}) or {},
                "reasoning": item.get("reasoning", ""),
                "error": item.get("error"),
            }

        state.decisions.last_founder_decision = agent_map.get("founder", {})
        state.decisions.last_marketing_decision = agent_map.get("marketing", {})
        state.decisions.last_investor_decision = agent_map.get("investor", {})

    def _update_hybrid_reputation(self, state):
        """
        Update reputation after the completed step has been recorded.
        Tries a few likely locations for the hybrid engine so this stays
        robust across wrapper/service layouts.
        """
        candidates = [
            self.deliberation_service,
            getattr(self.deliberation_service, "engine", None),
            getattr(self.deliberation_service, "hybrid_engine", None),
            getattr(self.deliberation_service, "deliberation_engine", None),
        ]

        for candidate in candidates:
            updater = getattr(candidate, "_update_reputation_from_previous_step", None)
            if callable(updater):
                try:
                    updater(state)
                except Exception:
                    pass
                return

    def _update_ppo_learning(self, prev_snapshot, decision, reward_total, new_state, deliberation_output, metrics):
        """
        Feed the completed transition into the PPO adapter's lightweight
        online learning layer.
        """
        engine = getattr(self.deliberation_service, "engine", None)
        ppo = getattr(engine, "ppo", None)
        if ppo is None:
            return

        try:
            ppo.record_transition(
                prev_snapshot,
                decision,
                reward_total,
                new_state,
                info={
                    "step": new_state.step,
                    "metrics": metrics,
                    "selected_agent": deliberation_output.get("selected_agent"),
                },
            )
            ppo.maybe_learn()
        except Exception:
            pass

    def execute_step(self, state):
        """
        Executes one full simulation step
        """
        # 1. Run hybrid deliberation via service (LLM agents + PPO weighting)
        deliberation_output = self.deliberation_service.run(state)
        decision = deliberation_output["final_decision"]

        # Snapshot pre-decision state for reward computation
        prev_snapshot = state.copy(deep=True)

        # 2. Apply decision to environment
        new_state = apply_market_dynamics(state, decision)

        # 3. Compute metrics
        metrics = compute_metrics(new_state)

        # 4. Compute multi-objective reward
        reward = compute_reward(prev_snapshot, new_state, metrics)
        self._logs_repo.save_log({
            "data": {
                "type": "hybrid_reward",
                "step": new_state.step,
                "reward": reward,
            }
        })

        # 5. Persist agent decisions for traceability
        self._store_agent_decisions(new_state, deliberation_output)

        # 6. Update history with richer debugging info
        new_state.history.append({
            "step": new_state.step,
            "decision": decision,
            "metrics": metrics,
            "reward": reward["total"],
            "reward_breakdown": reward["components"],
            "deliberation": {
                "weights": deliberation_output.get("weights", {}),
                "proposal_ranking": deliberation_output.get("proposal_ranking", []),
                "selected_agent": deliberation_output.get("selected_agent"),
                "trust_scores": deliberation_output.get("trust_scores", {}),
                "agent_stats": deliberation_output.get("agent_stats", {}),
                "ppo_available": deliberation_output.get("ppo_available", False),
                "ppo_reference_decision": deliberation_output.get("ppo_reference_decision"),
                "ppo_value_estimate": deliberation_output.get("ppo_value_estimate"),
            },
        })

        # 7. Update hybrid memory/reputation after the step is fully recorded
        self._update_hybrid_reputation(new_state)

        # 8. Feed transition to PPO online learning
        self._update_ppo_learning(
            prev_snapshot,
            decision,
            reward["total"],
            new_state,
            deliberation_output,
            metrics,
        )

        # 9. Increment step
        new_state.step += 1

        return new_state, deliberation_output, metrics