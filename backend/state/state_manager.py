# backend/state/state_manager.py

from backend.state.initializer import initialize_state
from backend.state.state_schema import SimulationState


class StateManager:
    def __init__(self):
        self._state: SimulationState = initialize_state()

    def get_state(self) -> SimulationState:
        """
        Return current state
        """
        return self._state

    def set_state(self, new_state: SimulationState):
        """
        Replace entire state (used after step execution)
        """
        self._state = new_state

    def reset(self):
        """
        Reset to initial state
        """
        self._state = initialize_state()

    def update_history(self, entry: dict):
        """
        Append to state history
        """
        self._state.history.append(entry)

    def to_dict(self) -> dict:
        """
        Return JSON-friendly state
        """
        return self._state.dict()