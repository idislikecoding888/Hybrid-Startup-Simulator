# backend/hybrid/ppo_adapter.py
#
# Thin wrapper around the supplied Stable-Baselines3 PPO checkpoint.
#
# IMPORTANT: PPO is not used here to pick a business decision directly.
# Its trained action head outputs 3 continuous values in [0,1] that map to
# this simulator's 3 decision variables (price, quality, marketing_budget).
# We reuse that trained output as a *reference decision* - i.e. "what a
# policy optimized purely for long-run reward would do from this state".
# weighting.py then measures how closely each LLM agent's proposal agrees
# with this reference to derive adaptive trust weights per agent. This is
# what turns PPO into a coordination/weighting mechanism instead of a
# competing decision-maker.
#
# Robustness: if the checkpoint can't be loaded or inference throws, every
# public method degrades to a neutral/no-op result and `available` is set
# to False so downstream code (weighting.py) can fall back to equal
# weights, per the spec.

import os
from typing import Dict, Optional

import numpy as np

from backend.hybrid.state_encoder import encode_state

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "checkpoint", "ppo_checkpoint.zip"
)

# Reference-decision output ranges. These reuse the same clamp ranges the
# existing (pre-hybrid) DeliberationEngine already validated its LLM output
# against, so the PPO reference and the LLM proposals live on the same
# scale.
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
            from stable_baselines3 import PPO  # local import: optional dep

            if not os.path.exists(checkpoint_path):
                return
            self.model = PPO.load(checkpoint_path, device="cpu")
            self.available = True
        except Exception:
            # Missing dependency, corrupt checkpoint, version mismatch, etc.
            # Caller must handle self.available == False.
            self.available = False
            self.model = None

    def reference_decision(self, state) -> Optional[Dict[str, float]]:
        """Returns PPO's implied {price, quality, marketing_budget} target,
        or None if PPO is unavailable / inference fails."""
        if not self.available:
            return None
        try:
            obs = encode_state(state)
            action, _ = self.model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            return {
                "price": _denorm(action[0], *PRICE_RANGE),
                "quality": _denorm(action[1], *QUALITY_RANGE),
                "marketing_budget": _denorm(action[2], *BUDGET_RANGE),
            }
        except Exception:
            return None

    def value_estimate(self, state) -> Optional[float]:
        """Critic's value estimate for the current state (used for logging /
        as an optional confidence signal). None if unavailable."""
        if not self.available:
            return None
        try:
            obs = encode_state(state)
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs.reshape(1, -1))
            value = self.model.policy.predict_values(obs_tensor)
            return float(value.detach().cpu().numpy().reshape(-1)[0])
        except Exception:
            return None
