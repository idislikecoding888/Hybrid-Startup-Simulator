from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import re

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

        # If the caller attached this agent's own memory block to the
        # context (see HybridDeliberationEngine._run_agents), prepend it to
        # the existing prompt. This reuses the same single LLM call below -
        # no extra API calls are introduced.
        memory_block = ""
        if context and isinstance(context, dict):
            memory_block = context.get("agent_memory_block", "") or ""

        if memory_block:

            instruction = """
        You are participating in a long-running startup simulation.

        The memory below contains ONLY your own previous decisions and outcomes.

        Treat it as your personal experience.

        Learn from strategies that produced higher rewards.

        Avoid repeating strategies that repeatedly resulted in poor outcomes.

        Use this memory to improve your next proposal, but still prioritize the CURRENT simulation state over older memories.

        ------------------------------------------------------------

        """

            prompt = instruction + memory_block + prompt

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
        from backend.agents.base.llm_interface import call_llm
        return call_llm(prompt)

    def _extract_json_candidate(self, text: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"```(?:json)?", "", text)
        text = text.replace("```", "").strip()

        # Try to isolate the first JSON object in the response
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0)

        return text

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Ensures output is valid JSON-like dict.
        """
        try:
            candidate = self._extract_json_candidate(response)
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                proposal = parsed.get("proposal")
                if isinstance(proposal, dict) and "confidence" not in proposal:
                    proposal["confidence"] = 0.6
                return parsed

        except Exception:
            pass

        return {
            "agent": self.name,
            "error": "Invalid JSON response",
            "raw_output": response
        }