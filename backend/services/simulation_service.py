# backend/services/simulation_service.py

from backend.engine.simulation_engine import SimulationEngine
from backend.agents.founder_agent import FounderAgent
from backend.agents.marketing_agent import MarketingAgent
from backend.agents.investor_agent import InvestorAgent
from backend.agents.customer_agent import CustomerAgent
from backend.services.deliberation_service import DeliberationService


class SimulationService:
    def __init__(self):
        self.running = False
        self.agents = self._init_agents()
        self.deliberation_service = DeliberationService(self.agents)
        # Pass deliberation_service into the engine so StepExecutor uses it
        self.engine = SimulationEngine(self.agents, self.deliberation_service)

    def start(self, steps: int = 10):
        if self.running:
            return {"error": "Simulation already running"}

        self.running = True  # FIX: was missing — guard never blocked anything
        try:
            result = self.engine.run(steps)
            return result
        finally:
            self.running = False

    def _init_agents(self):
        return [
            FounderAgent(),
            MarketingAgent(),
            InvestorAgent(),
            CustomerAgent(),
        ]

    def step(self):
        return self.engine.step()

    def reset(self):
        self.engine.reset()
        return {"status": "reset"}

    def stop(self):
        self.engine.stop()
        return {"status": "stopped"}

    def get_state(self):
        return self.engine.state.dict()
