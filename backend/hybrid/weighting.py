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
    return _clip01(conf) if conf is not None else 0.6


def _safe_inventory_stock(state) -> float:
    inventory = getattr(state, "inventory", None)
    if isinstance(inventory, dict):
        return float(inventory.get("stock", 0) or 0)

    stock = getattr(inventory, "stock", None)
    if stock is None:
        return 0.0

    try:
        return float(stock)
    except (TypeError, ValueError):
        return 0.0


def _reputation_score(agent_name: str, agent_reputation: Optional[Dict[str, float]]) -> float:
    if not agent_reputation:
        return 0.5
    return _clip01(agent_reputation.get(agent_name, 0.5))


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

    cash = max(float(state.finance.cash), 1.0)
    ppo_risk = _clip01(ref["marketing_budget"] / cash)

    err = abs(investor_risk - ppo_risk)
    score = 1.0 - err
    if approve is False:
        score *= 0.7
    return score


def _customer_alignment(output: Dict, ref: Dict) -> float:
    p = output.get("proposal", {}) or {}
    feedback = p.get("feedback_score")
    if feedback is None:
        return 0.0

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


def _proposal_feasibility(output: Dict, state) -> float:
    agent = output.get("agent", "")
    proposal = output.get("proposal", {}) or {}
    score = 1.0

    if agent == "founder":
        price = proposal.get("price")
        quality = proposal.get("quality")
        if price is None or quality is None:
            return 0.0
        price = float(price)
        quality = float(quality)

        if not (100.0 <= price <= 2000.0):
            score *= 0.2
        if not (0.1 <= quality <= 1.0):
            score *= 0.2

        if price > state.product.price * 1.6 and state.customers.satisfaction < 0.8:
            score *= 0.85
        if quality < state.product.quality - 0.15 and state.customers.satisfaction < 0.7:
            score *= 0.85

    elif agent == "marketing":
        budget = proposal.get("marketing_budget")
        if budget is None:
            return 0.0
        budget = float(budget)
        cash = max(float(state.finance.cash), 1.0)

        if budget < 0 or budget > 5000:
            score *= 0.2
        if budget > cash * 0.35:
            score *= 0.85
        if _safe_inventory_stock(state) < 100 and budget > state.marketing.budget:
            score *= 0.9

    elif agent == "investor":
        approve = proposal.get("approve", True)
        risk = str(proposal.get("risk_level", "medium")).lower()
        if approve is False and state.finance.cash > 0:
            score *= 0.95
        risk_map = {"low": 1.0, "medium": 0.95, "high": 0.85}
        score *= risk_map.get(risk, 0.9)

    elif agent == "customer":
        feedback = proposal.get("feedback_score")
        sentiment = str(proposal.get("sentiment", "neutral")).lower()
        if feedback is None:
            return 0.0

        feedback = _clip01(feedback)
        expected = _clip01(
            0.6
            + (state.product.quality - 0.5)
            - max(0.0, (state.product.price - 500.0) / 1000.0)
        )
        consistency = 1.0 - min(abs(feedback - expected), 1.0)
        score *= 0.5 + 0.5 * consistency

        if sentiment == "negative" and state.customers.satisfaction > 0.75:
            score *= 0.9

    return _clip01(score)


def _score_output(
    output: Dict,
    reference_decision: Optional[Dict],
    state,
    agent_reputation: Optional[Dict[str, float]],
    agent_stats=None,
) -> Dict:
    agent_name = output.get("agent")
    confidence = _agent_confidence(output)
    feasibility = _proposal_feasibility(output, state)
    reputation = _reputation_score(agent_name, agent_reputation)
    stats = (agent_stats or {}).get(agent_name, {})

    historical_reward = _clip01(
        stats.get("average_reward", 0.5)
    )

    historical_success = min(
        stats.get("successful_decisions", 0) / 10.0,
        1.0,
    )

    if reference_decision is None or not getattr(settings, "HYBRID_PPO_ENABLED", True):
        alignment = 0.5
        utility = (
            0.34 * confidence
            + 0.28 * feasibility
            + 0.18 * reputation
            + 0.12 * historical_reward
            + 0.08 * historical_success
        )
    else:
        alignment_fn = _ALIGNMENT_FUNCS.get(agent_name)
        alignment = alignment_fn(output, reference_decision, state) if alignment_fn else 0.5
        utility = (
            0.28 * alignment
            + 0.20 * confidence
            + 0.18 * feasibility
            + 0.16 * reputation
            + 0.10 * historical_reward
            + 0.08 * historical_success
        )

    if agent_name == "founder" and state.customers.satisfaction < 0.7:
        utility += 0.03
    elif agent_name == "marketing" and state.finance.cash > 100000:
        utility += 0.02
    elif agent_name == "investor" and state.finance.cash < 50000:
        utility += 0.03
    elif agent_name == "customer" and state.product.price > 500:
        utility += 0.02

    utility = _clip01(utility)

    return {
        "agent": agent_name,
        "proposal": output.get("proposal", {}) or {},
        "reasoning": output.get("reasoning", ""),
        "error": output.get("error"),
        "alignment": float(_clip01(alignment)),
        "confidence": float(confidence),
        "feasibility": float(feasibility),
        "reputation": float(reputation),
        "utility": float(utility),
        "selection_reason": (
            f"utility={utility:.3f} "
            f"(alignment={alignment:.3f}, confidence={confidence:.3f}, "
            f"feasibility={feasibility:.3f}, reputation={reputation:.3f})"
        ),
    }


def rank_agent_proposals(
    agent_outputs: List[Dict],
    reference_decision: Optional[Dict],
    state,
    agent_reputation: Optional[Dict[str, float]] = None,
    agent_stats=None,
) -> List[Dict]:
    """
    Returns a ranked list of proposal metadata objects sorted by utility.
    The top-ranked proposal is marked selected=True.
    """
    usable = [o for o in agent_outputs if _has_usable_proposal(o)]
    if not usable:
        return []

    ranked = [
        _score_output(
            output,
            reference_decision,
            state,
            agent_reputation,
            agent_stats,
        )
        for output in usable
    ]
    ranked.sort(key=lambda x: x["utility"], reverse=True)

    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1
        item["selected"] = idx == 0

    return ranked


def compute_agent_weights(
    agent_outputs: List[Dict],
    reference_decision: Optional[Dict],
    state,
    agent_reputation: Optional[Dict[str, float]] = None,
    agent_stats=None,
) -> Dict[str, float]:
    """
    Returns {agent_name: weight}, weights sum to 1.

    - If PPO is unavailable, falls back to equal weights across surviving agents.
    - If no usable proposals exist, returns {}.
    """
    usable = [o for o in agent_outputs if _has_usable_proposal(o)]
    if not usable:
        return {}

    names = [o["agent"] for o in usable]

    ppo_enabled = getattr(settings, "HYBRID_PPO_ENABLED", True)
    if reference_decision is None or not ppo_enabled:
        w = 1.0 / len(names)
        return {name: w for name in names}

    ranked = rank_agent_proposals(
        agent_outputs,
        reference_decision,
        state,
        agent_reputation=agent_reputation,
        agent_stats=agent_stats,
    )
    if not ranked:
        return {}

    scores = np.array([item["utility"] for item in ranked], dtype=np.float64)

    logits = scores / TEMPERATURE
    logits -= logits.max()
    exp = np.exp(logits)
    weights = exp / exp.sum()

    return {item["agent"]: float(w) for item, w in zip(ranked, weights)}