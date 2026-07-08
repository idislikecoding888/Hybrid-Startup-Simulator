# backend/engine/simulation_engine.py

from typing import List

from backend.engine.step_executor import StepExecutor
from backend.state.initializer import initialize_state
from backend.config import settings


class SimulationEngine:
    def __init__(self, agents: List, deliberation_service):
        self.agents = agents
        # FIX: pass deliberation_service down to StepExecutor
        self.executor = StepExecutor(deliberation_service)
        self.state = initialize_state()
        self.running = False

    def reset(self):
        self.state = initialize_state()

        # Also reset per-agent memory (isolated Founder/Marketing/Investor/
        # Customer histories) so it doesn't leak across simulation resets.
        deliberation_service = getattr(self.executor, "deliberation_service", None)
        hybrid_engine = getattr(deliberation_service, "engine", None)
        reset_memory = getattr(hybrid_engine, "reset_memory", None)
        if callable(reset_memory):
            reset_memory()

    def run(self, steps: int = None):
        if steps is None:
            steps = settings.MAX_STEPS

        self.running = True
        results = []

        for _ in range(steps):
            if not self.running:
                break

            self.state, deliberation_output, metrics = self.executor.execute_step(self.state)

            results.append({
                "step": self.state.step,
                "state": self.state.dict(),
                "metrics": metrics,
                "deliberation": deliberation_output,
            })

        self.running = False
        return results

    def step(self):
        self.state, deliberation_output, metrics = self.executor.execute_step(self.state)
        return {
            "step": self.state.step,
            "state": self.state.dict(),
            "metrics": metrics,
            "deliberation": deliberation_output,
        }

    def stop(self):
        self.running = False
