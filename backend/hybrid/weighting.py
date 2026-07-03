# backend/hybrid/weighting.py
#
# Turns (agent proposals, PPO reference decision, current state) into a
# normalized dict of per-agent trust weights.
#
# Method: for each agent we compute an "alignment" score in [0,1] that
# measures how consistent that agent's proposal is with PPO's reference
# decision (the policy's learned notion of a good move from this state),
# scaled by the agent's own self-reported confidence. Alignment scores are
# then passed through a softmax so weights are smooth, always positive and
# sum to 1 - this is the "PPO decides how much to listen to each agent"
# mechanism requested, without PPO ever emitting a business decision
# itself.

from typing import Dict, List, Optional

import numpy as np

from backend.config import settings

TEMPERATURE = getattr(settings, "HYBRID_WEIGHTING_TEMPERATURE", 0.5)


def _clip01(x) -> float:
    try:
        return float(np.clip(float(x), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.5


def _agent_confidence(output: Dict) -> float:
    proposal = output.get("proposal", {}) if isinstance(output, dict) else {}
    conf = None
    if isinstance(proposal, dict):
        conf = proposal.get("confidence")
    if conf is None and isinstance(output, dict):
        conf = output.get("confidence")
    return _clip01(conf) if conf is not None else 0.6  # neutral default


def _founder_alignment(output: Dict, ref: Dict) -> float:
    p = output.get("proposal", {}) or {}
    price = p.get("price")
    quality = p.get("quality")
    if price is None and quality is None:
        return 0.0
    price_err = abs(float(price) - ref["price"]) / (ref["price"] + 1e-6) if price is not None else 0.0
    quality_err = abs(float(quality) - ref["quality"]) if quality is not None else 0.0
    err = 0.5 * min(price_err, 2.0) / 2.0 + 0.5 * min(quality_err, 1.0)
    return 1.0 - err


def _marketing_alignment(output: Dict, ref: Dict) -> float:
    p = output.get("proposal", {}) or {}
    budget = p.get("marketing_budget")
    if budget is None:
        return 0.0
    err = abs(float(budget) - ref["marketing_budget"]) / (ref["marketing_budget"] + 1e-6)
    return 1.0 - min(err, 2.0) / 2.0


def _investor_alignment(output: Dict, ref: Dict, state) -> float:
    p = output.get("proposal", {}) or {}
    approve = p.get("approve")
    risk_level = str(p.get("risk_level", "medium")).lower()
    risk_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
    investor_risk = risk_map.get(risk_level, 0.5)

    # PPO-implied risk appetite: how aggressive is the reference spend
    # relative to available cash.
    cash = max(float(state.finance.cash), 1.0)
    ppo_risk = _clip01(ref["marketing_budget"] / cash)

    err = abs(investor_risk - ppo_risk)
    score = 1.0 - err
    if approve is False:
        score *= 0.7  # a veto still counts, but tempers influence a bit
    return score


def _customer_alignment(output: Dict, ref: Dict) -> float:
    p = output.get("proposal", {}) or {}
    feedback = p.get("feedback_score")
    if feedback is None:
        return 0.0
    # Simple utility heuristic for "expected satisfaction" under PPO's
    # reference price/quality, reusing the same shape as the satisfaction
    # formula in environment/market_model.py.
    price_penalty = max(0.0, (ref["price"] - 500.0) / 1000.0)
    quality_boost = ref["quality"] - 0.5
    predicted = _clip01(0.6 + quality_boost - price_penalty)
    err = abs(float(feedback) - predicted)
    return 1.0 - min(err, 1.0)


_ALIGNMENT_FUNCS = {
    "founder": lambda out, ref, state: _founder_alignment(out, ref),
    "marketing": lambda out, ref, state: _marketing_alignment(out, ref),
    "investor": lambda out, ref, state: _investor_alignment(out, ref, state),
    "customer": lambda out, ref, state: _customer_alignment(out, ref),
}


def _has_usable_proposal(output: Dict) -> bool:
    return (
        isinstance(output, dict)
        and isinstance(output.get("proposal"), dict)
        and len(output["proposal"]) > 0
        and not output.get("error")
    )


def compute_agent_weights(
    agent_outputs: List[Dict],
    reference_decision: Optional[Dict],
    state,
) -> Dict[str, float]:
    """
    agent_outputs: list of {"agent": name, "proposal": {...}, "reasoning": str}
    reference_decision: PPO's {price, quality, marketing_budget} target, or
        None if PPO is unavailable.
    Returns: {agent_name: weight}, weights sum to 1.

    Robustness:
    - Agents with no usable proposal (failed/crashed) are excluded and
      their weight is implicitly redistributed among the rest (softmax
      renormalizes over the surviving set).
    - If PPO is unavailable, falls back to equal weights across surviving
      agents, per spec.
    """
    usable = [o for o in agent_outputs if _has_usable_proposal(o)]
    if not usable:
        return {}

    names = [o["agent"] for o in usable]

    ppo_enabled = getattr(settings, "HYBRID_PPO_ENABLED", True)
    if reference_decision is None or not ppo_enabled:
        w = 1.0 / len(names)
        return {name: w for name in names}

    scores = []
    for output in usable:
        agent_name = output["agent"]
        alignment_fn = _ALIGNMENT_FUNCS.get(agent_name)
        alignment = alignment_fn(output, reference_decision, state) if alignment_fn else 0.5
        confidence = _agent_confidence(output)
        scores.append(alignment * confidence)

    scores = np.array(scores, dtype=np.float64)
    logits = scores / TEMPERATURE
    logits -= logits.max()  # numerical stability
    exp = np.exp(logits)
    weights = exp / exp.sum()

    return {name: float(w) for name, w in zip(names, weights)}
