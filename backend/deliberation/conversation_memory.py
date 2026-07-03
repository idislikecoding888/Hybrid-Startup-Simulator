# backend/deliberation/conversation_memory.py

from typing import List, Dict, Optional


# ---------------- MEMORY LAYER ---------------- #

class ConversationMemory:
    def __init__(self):
        self.history: List[Dict] = []

    def add_entry(self, entry: Dict):
        self.history.append(entry)

    def get_all(self) -> List[Dict]:
        return self.history

    def get_latest(self) -> Dict:
        return self.history[-1] if self.history else {}

    def get_by_type(self, entry_type: str) -> List[Dict]:
        return [entry for entry in self.history if entry.get("type") == entry_type]

    def get_last_n(self, n: int) -> List[Dict]:
        return self.history[-n:]

    def clear(self):
        self.history = []


# ---------------- MANAGER LAYER ---------------- #

class ConversationManager:
    """
    High-level interface for managing deliberation logs.
    """

    def __init__(self, memory: Optional[ConversationMemory] = None):
        self.memory = memory if memory else ConversationMemory()

    def log_proposal(self, proposal: Dict):
        self.memory.add_entry({
            "type": "proposal",
            "proposal": proposal
        })

    def log_critique(self, critique: Dict):
        self.memory.add_entry({
            "type": "critique",
            "critique": critique
        })

    def log_negotiation(self, negotiation: Dict):
        self.memory.add_entry({
            "type": "negotiation",
            "negotiation": negotiation
        })

    def log_consensus(self, consensus: Dict):
        self.memory.add_entry({
            "type": "consensus",
            "consensus": consensus
        })

    def get_full_conversation(self) -> List[Dict]:
        return self.memory.get_all()

    def get_latest(self) -> Dict:
        return self.memory.get_latest()

    def get_by_type(self, entry_type: str) -> List[Dict]:
        return self.memory.get_by_type(entry_type)

    def get_last_n(self, n: int) -> List[Dict]:
        return self.memory.get_last_n(n)

    def clear(self):
        self.memory.clear()