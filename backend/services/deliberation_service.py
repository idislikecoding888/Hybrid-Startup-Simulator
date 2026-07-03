# backend/services/deliberation_service.py

from typing import List, Dict
from backend.hybrid.hybrid_engine import HybridDeliberationEngine
from backend.deliberation.conversation_memory import ConversationMemory
from backend.storage.logs_repo import LogsRepository


class DeliberationService:
    """
    Service layer: routes execution through the hybrid LLM + PPO engine
    (every LLM agent runs, PPO produces adaptive per-agent weights, the
    proposals are fused using those weights), handles logging, and exposes
    log-fetch methods.

    The old multi-agent managers (ProposalManager, CritiqueManager,
    NegotiationManager, ConsensusManager) remain unused/removed from the
    active path, as before. The previous single-call DeliberationEngine is
    still available at backend/deliberation/deliberation_engine.py for
    reference/rollback but is no longer wired in here.
    """

    def __init__(self, agents: List):
        self.agents = agents
        self.engine = HybridDeliberationEngine(agents)
        self.memory = ConversationMemory()
        self.logs_repo = LogsRepository()

    # ---------- MAIN EXECUTION ---------- #

    def run(self, state) -> Dict:
        """
        Execute one hybrid deliberation step (LLM agents + PPO weighting).
        Logs agent outputs, PPO weights, and the fused decision to memory + DB.
        """
        result = self.engine.run_deliberation(state)

        entry = {
            "type": "hybrid_deliberation",
            "agent_outputs": result.get("agent_outputs", []),
            "ppo_available": result.get("ppo_available", False),
            "ppo_reference_decision": result.get("ppo_reference_decision"),
            "ppo_value_estimate": result.get("ppo_value_estimate"),
            "weights": result.get("weights", {}),
            "final_decision": result.get("final_decision", {}),
            "reasoning": result.get("reasoning", ""),
        }

        # Log to in-memory store
        self.memory.add_entry(entry)

        # FIX: logs_repo.save_log() expects {"data": {...}} as the wrapper,
        # not the log dict directly. Wrapping correctly here.
        self.logs_repo.save_log({"data": entry})

        return result

    # ---------- FETCH METHODS ---------- #

    def get_logs(self) -> List[Dict]:
        return self.logs_repo.get_all_logs()

    def get_latest_log(self) -> Dict:
        return self.logs_repo.get_latest_log()

    def get_logs_by_type(self, log_type: str) -> List[Dict]:
        return self.logs_repo.get_logs_by_type(log_type)

    def get_last_n_logs(self, n: int) -> List[Dict]:
        return self.logs_repo.get_last_n_logs(n)

    # ---------- CONTROL ---------- #

    def clear_logs(self):
        self.memory.clear()
        self.logs_repo.clear_logs()

    # ---------- SNAPSHOT FOR FRONTEND ---------- #

    def get_conversation_snapshot(self) -> Dict:
        entries = self.memory.get_all()
        return {
            "total_steps": len(entries),
            "history": entries,
        }
