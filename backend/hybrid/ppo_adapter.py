import json
import os
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from backend.hybrid.state_encoder import encode_state

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoint",
    "ppo_checkpoint.zip",
)

ONLINE_BIAS_PATH = os.path.join(
    os.path.dirname(__file__),
    "checkpoint",
    "ppo_online_bias.json",
)

PRICE_RANGE = (100.0, 2000.0)
QUALITY_RANGE = (0.1, 1.0)
BUDGET_RANGE = (100.0, 5000.0)


def _denorm(x: float, lo: float, hi: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return lo + x * (hi - lo)


def _norm(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


class PPOAdapter:
    """
    Inference wrapper + lightweight online adaptation layer.

    The supplied PPO checkpoint is still used for inference, but after each
    simulation step we can record the executed transition and update a small
    bias vector from reward-weighted experience. This gives the policy a
    practical learning loop without requiring a missing standalone trainer.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self.available = False
        self.model = None

        # Lightweight online learning state.
        self._transition_buffer: Deque[Dict] = deque(maxlen=2048)
        self._policy_bias = np.zeros(3, dtype=np.float32)
        self._update_counter = 0

        self._load(checkpoint_path)
        self._load_online_bias()

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

    def _load_online_bias(self) -> None:
        try:
            if not os.path.exists(ONLINE_BIAS_PATH):
                return
            with open(ONLINE_BIAS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            bias = np.asarray(data.get("policy_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
            if bias.shape == (3,):
                self._policy_bias = np.clip(bias, -0.35, 0.35)
        except Exception:
            pass

    def _save_online_bias(self) -> None:
        try:
            os.makedirs(os.path.dirname(ONLINE_BIAS_PATH), exist_ok=True)
            with open(ONLINE_BIAS_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {"policy_bias": self._policy_bias.astype(float).tolist()},
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def _safe_action(self, action) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 3:
            padded = np.zeros(3, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        elif action.size > 3:
            action = action[:3]
        return np.clip(action, 0.0, 1.0)

    def _apply_online_bias(self, action: np.ndarray) -> np.ndarray:
        # Small bias only; keep the checkpoint as the main policy.
        return np.clip(action + 0.25 * self._policy_bias, 0.0, 1.0)

    def _decision_to_action(self, decision: Dict[str, float]) -> np.ndarray:
        return np.asarray(
            [
                _norm(decision.get("price", 0.0), *PRICE_RANGE),
                _norm(decision.get("quality", 0.0), *QUALITY_RANGE),
                _norm(decision.get("marketing_budget", 0.0), *BUDGET_RANGE),
            ],
            dtype=np.float32,
        )

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
            action = self._apply_online_bias(action)

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

    def record_transition(
        self,
        prev_state,
        executed_decision: Dict[str, float],
        reward: float,
        next_state,
        info: Optional[Dict] = None,
    ) -> None:
        """
        Store one completed simulation transition for reward-weighted online
        adaptation. This does not call the LLM and does not change the API
        contract.
        """
        try:
            transition = {
                "state": encode_state(prev_state).astype(np.float32),
                "action": self._decision_to_action(executed_decision),
                "reward": float(reward),
                "next_state": encode_state(next_state).astype(np.float32),
                "done": bool((info or {}).get("done", False)),
            }
            self._transition_buffer.append(transition)
        except Exception:
            pass

    def maybe_learn(
        self,
        min_batch_size: int = 4,
        learn_every: int = 1,
        max_batch_size: int = 64,
    ) -> bool:
        """
        Lightweight online adaptation:
        - reward-weighted updates over recent executed decisions
        - updates only a tiny bias vector, keeping the checkpoint intact

        This is intentionally conservative because the zip does not contain
        a dedicated PPO trainer/environment loop.
        """
        if len(self._transition_buffer) < min_batch_size:
            return False

        self._update_counter += 1
        if learn_every > 1 and (self._update_counter % learn_every) != 0:
            return False

        try:
            batch = list(self._transition_buffer)[-min(len(self._transition_buffer), max_batch_size):]
            rewards = np.asarray([t["reward"] for t in batch], dtype=np.float32)
            actions = np.asarray([t["action"] for t in batch], dtype=np.float32)

            if rewards.size == 0 or actions.size == 0:
                return False

            finite_mask = np.isfinite(rewards).reshape(-1)
            if not finite_mask.any():
                return False

            rewards = rewards[finite_mask]
            actions = actions[finite_mask]

            # Turn reward into positive weights.
            centered = rewards - rewards.mean()
            scale = rewards.std() + 1e-6
            signal = np.tanh(centered / scale)

            weights = signal - signal.min() + 1e-3
            total = float(weights.sum())
            if total <= 0:
                weights = np.ones_like(weights, dtype=np.float32) / float(len(weights))
            else:
                weights = weights / total

            target_action = np.average(actions, axis=0, weights=weights)
            target_bias = np.clip(target_action - 0.5, -0.35, 0.35).astype(np.float32)

            # Small, stable update.
            self._policy_bias = np.clip(
                0.90 * self._policy_bias + 0.10 * target_bias,
                -0.35,
                0.35,
            )
            self._save_online_bias()
            return True
        except Exception:
            return False

    def learning_summary(self) -> Dict:
        """
        Returns diagnostics so we can verify PPO is
        actually learning during the simulation.
        """

        last_reward = None

        if len(self._transition_buffer) > 0:
            last_reward = float(self._transition_buffer[-1]["reward"])

        return {

            "available": self.available,

            "buffer_size": len(self._transition_buffer),

            "updates": self._update_counter,

            "last_reward": last_reward,

            "policy_bias": {

                "price": round(float(self._policy_bias[0]), 5),

                "quality": round(float(self._policy_bias[1]), 5),

                "marketing_budget": round(float(self._policy_bias[2]), 5)

            },

            "learning_triggered":

                len(self._transition_buffer) >= 4

        }