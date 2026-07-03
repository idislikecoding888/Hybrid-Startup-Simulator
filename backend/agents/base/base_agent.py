# backend/agents/base/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any

from backend.state.state_schema import SimulationState


class BaseAgent(ABC):
    def __init__(self, name: str, goal: str):
        self.name = name
        self.goal = goal

    def run(self, state: SimulationState, context: Dict = None) -> Dict:
        """
        Main entry point for agent execution.
        """
        prompt = self.build_prompt(state, context)
        response = self.call_llm(prompt)
        parsed = self.parse_response(response)
        return parsed

    @abstractmethod
    def build_prompt(self, state: SimulationState, context: Dict = None) -> str:
        """
        Each agent defines its own prompt.
        """
        pass

    def call_llm(self, prompt: str) -> str:
        """
        Calls LLM (will be implemented via llm_interface later)
        """
        from backend.agents.base.llm_interface import call_llm

        return call_llm(prompt)

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Ensures output is valid JSON-like dict.
        """
        import json

        try:
            parsed = json.loads(response)
            proposal = parsed.get("proposal")
            if isinstance(proposal, dict) and "confidence" not in proposal:
                # Estimate a confidence score if the LLM didn't provide one,
                # so downstream PPO weighting always has a signal to use.
                proposal["confidence"] = 0.6
            return parsed
        except Exception:
            return {
                "agent": self.name,
                "error": "Invalid JSON response",
                "raw_output": response
            }