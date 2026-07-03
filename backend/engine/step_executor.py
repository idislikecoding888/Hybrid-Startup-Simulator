# backend/engine/step_executor.py

from backend.environment.market_model import apply_market_dynamics
from backend.metrics.metrics_engine import compute_metrics
from backend.hybrid.reward import compute_reward
from backend.storage.logs_repo import LogsRepository


class StepExecutor:
    def __init__(self, deliberation_service):
        # FIX: inject DeliberationService instead of creating DeliberationEngine directly
        self.deliberation_service = deliberation_service
        self._logs_repo = LogsRepository()

    def execute_step(self, state):
        """
        Executes one full simulation step
        """
        # 1. Run hybrid deliberation via service (LLM agents + PPO weighting)
        deliberation_output = self.deliberation_service.run(state)
        decision = deliberation_output["final_decision"]

        # Snapshot pre-decision state for reward computation -- apply_market_dynamics
        # mutates `state` in place, so the original values would otherwise be lost.
        prev_snapshot = state.copy(deep=True)

        # 2. Apply decision to environment
        new_state = apply_market_dynamics(state, decision)

        # 3. Compute metrics
        metrics = compute_metrics(new_state)

        # Multi-objective reward for the PPO weighting layer (logged now for
        # research analysis; consumed directly if/when training is enabled).
        reward = compute_reward(prev_snapshot, new_state, metrics)
        self._logs_repo.save_log({"data": {
            "type": "hybrid_reward",
            "step": new_state.step,
            "reward": reward,
        }})

        # 4. Update history
        new_state.history.append({
            "step": new_state.step,
            "decision": decision,
            "metrics": metrics,
            "reward": reward["total"],
        })

        # 5. Increment step
        new_state.step += 1

        return new_state, deliberation_output, metrics
