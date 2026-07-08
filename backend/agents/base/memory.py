# backend/agents/base/memory.py

import json
from typing import List, Dict, Optional


class Memory:
    def __init__(self):
        self.history: List[Dict] = []

    def add(self, entry: Dict):
        self.history.append(entry)

    def get_recent(self, n: int = 5) -> List[Dict]:
        return self.history[-n:]

    def get_all(self) -> List[Dict]:
        return self.history


class AgentMemoryManager:
    """
    Lightweight, per-agent memory used to give each LLM agent awareness of
    its own recent decisions and how they played out.

    - Each agent keeps its own isolated list of entries (no sharing).
    - Only the latest MAX_ENTRIES raw entries are kept; anything older is
      folded into a compact rolling summary string so prompts stay small.
    - No LLM calls are made here - this only builds a text block that gets
      prepended to the existing prompt before the existing LLM call.
    """

    MAX_ENTRIES = 8
    # Rough budget so the injected block stays well under ~500 tokens
    # (~4 chars/token heuristic).
    MAX_PROMPT_CHARS = 1800
    MAX_SUMMARY_CHARS = 400

    def __init__(self, agent_names: Optional[List[str]] = None):
        self._entries: Dict[str, List[Dict]] = {}
        self._summaries: Dict[str, str] = {}
        for name in (agent_names or []):
            self._ensure(name)

    def _ensure(self, agent_name: str) -> None:
        if agent_name not in self._entries:
            self._entries[agent_name] = []
        if agent_name not in self._summaries:
            self._summaries[agent_name] = ""

    @staticmethod
    def _round(value) -> Optional[float]:
        try:
            if value is None:
                return None
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None
        

    def diagnostics(self):
        result = {}

        for agent in self._entries.keys():

            result[agent] = {

                "entries": len(self._entries[agent]),

                "summary": self._summaries.get(agent, ""),

                "recent": self._entries[agent][-3:],

            }

        return result

    def _auto_lesson(self, selected: bool, reward) -> str:
        reward_val = self._round(reward)
        if selected and reward_val is not None and reward_val >= 0.6:
            return "This approach was selected and produced a strong outcome."
        if selected:
            return "This approach was selected but the outcome was modest."
        if reward_val is not None and reward_val < 0.4:
            return "This proposal was not selected amid a weak overall outcome."
        return "This proposal was not selected this round."

    def add_entry(
        self,
        agent_name: str,
        step,
        proposal: Optional[Dict],
        selected: bool,
        reward,
        reputation,
        trust_score,
        lesson: Optional[str] = None,
    ) -> None:
        self._ensure(agent_name)

        entry = {
            "step": step,
            "proposal": proposal or {},
            "selected": bool(selected),
            "reward": self._round(reward),
            "reputation": self._round(reputation),
            "trust_score": self._round(trust_score),
            "lesson": lesson or self._auto_lesson(selected, reward),
        }

        entries = self._entries[agent_name]
        entries.append(entry)

        while len(entries) > self.MAX_ENTRIES:
            oldest = entries.pop(0)
            self._fold_into_summary(agent_name, oldest)

    def _fold_into_summary(self, agent_name: str, entry: Dict) -> None:
        outcome = "selected" if entry.get("selected") else "not selected"
        note = f"Step {entry.get('step')}: {outcome}, reward={entry.get('reward')}."
        prev = self._summaries.get(agent_name, "")
        combined = f"{prev} {note}".strip() if prev else note
        if len(combined) > self.MAX_SUMMARY_CHARS:
            combined = combined[-self.MAX_SUMMARY_CHARS:]
        self._summaries[agent_name] = combined

    @staticmethod
    def _format_entry(e: Dict) -> str:
        return (
            f"Step: {e.get('step')}\n"
            f"Proposal: {json.dumps(e.get('proposal', {}), ensure_ascii=False)}\n"
            f"Selected: {e.get('selected')}\n"
            f"Reward: {e.get('reward')}\n"
            f"Reputation: {e.get('reputation')}\n"
            f"Trust Score: {e.get('trust_score')}\n"
            f"Lesson Learned: {e.get('lesson')}\n"
        )

    def get_prompt_block(self, agent_name: str) -> str:
        """
        Returns a compact text block summarizing this agent's own memory,
        meant to be prepended to the existing prompt. Returns "" when there
        is nothing to add yet (first step).
        """
        self._ensure(agent_name)
        entries = list(self._entries[agent_name])
        summary = self._summaries.get(agent_name, "")

        if not entries and not summary:
            return ""

        def build(entries_subset):
            lines = ["AGENT_MEMORY (your own past decisions, most recent last):"]
            if summary:
                lines.append(f"Earlier history summary: {summary}")
            for e in entries_subset:
                lines.append(self._format_entry(e))
            return "\n".join(lines) + "\n\n"

        block = build(entries)

        # Enforce a soft token/char budget by trimming oldest raw entries
        # first (summary is already compact and always kept).
        while len(block) > self.MAX_PROMPT_CHARS and len(entries) > 1:
            entries = entries[1:]
            block = build(entries)

        return block

    def reset(self, agent_name: str) -> None:
        self._entries[agent_name] = []
        self._summaries[agent_name] = ""

    def reset_all(self) -> None:
        for name in list(self._entries.keys()):
            self.reset(name)
