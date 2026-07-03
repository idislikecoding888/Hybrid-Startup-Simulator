# backend/agents/base/memory.py

from typing import List, Dict


class Memory:
    def __init__(self):
        self.history: List[Dict] = []

    def add(self, entry: Dict):
        self.history.append(entry)

    def get_recent(self, n: int = 5) -> List[Dict]:
        return self.history[-n:]

    def get_all(self) -> List[Dict]:
        return self.history