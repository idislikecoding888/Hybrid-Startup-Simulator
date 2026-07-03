#
# Thin wrapper around the supplied Stable-Baselines3 PPO checkpoint.
#
# PPO is used as a reference policy / coordination signal. It outputs 3
# continuous values in [0,1] that map to:
#   price, quality, marketing_budget
#
# The hybrid layer can then use that signal for trust weighting or logging.

import os
from typing import Dict, Optional

import numpy as np

from backend.hybrid.state_encoder import encode_state

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoint",
    "ppo_checkpoint.zip"
)

PRICE_RANGE = (100.0, 2000.0)
QUALITY_RANGE = (0.1, 1.0)
BUDGET_RANGE = (100.0, 5000.0)


def _denorm(x: float, lo: float, hi: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return lo + x * (hi - lo)


class PPOAdapter:
    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self.available = False
        self.model = None
        self._load(checkpoint_path)

    def _load(self, checkpoint_path: str):
        try:
            from stable_baselines3 import PPO  # optional dependency

            if not os.path.exists(checkpoint_path):
                self.available = False
                self.model = None
                return

            self.model = PPO.load(checkpoint_path, device="cpu")
            self.available = True
        except Exception:
            self.available = False
            self.model = None

    def _safe_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        elif action.size > 3:
            action = action[:3]
        return np.clip(action, 0.0, 1.0)

    def reference_decision(self, state) -> Optional[Dict[str, float]]:
        """
        Returns PPO's implied {price, quality, marketing_budget} target,
        or None if PPO is unavailable / inference fails.
        """
        if not self.available:
            return None

        try:
            obs = encode_state(state)
            action, _ = self.model.predict(obs, deterministic=True)
            action = self._safe_action(action)

            return {
                "price": _denorm(action[0], *PRICE_RANGE),
                "quality": _denorm(action[1], *QUALITY_RANGE),
                "marketing_budget": _denorm(action[2], *BUDGET_RANGE),
            }
        except Exception:
            return None

    def value_estimate(self, state) -> Optional[float]:
        """
        Critic value estimate for the current state.
        """
        if not self.available:
            return None

        try:
            obs = encode_state(state)
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs.reshape(1, -1))
            value = self.model.policy.predict_values(obs_tensor)
            return float(value.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            return None