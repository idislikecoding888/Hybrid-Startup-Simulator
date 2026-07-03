from typing import Dict, List, Optional

PRICE_RANGE = (100.0, 2000.0)
QUALITY_RANGE = (0.1, 1.0)
BUDGET_RANGE = (100.0, 5000.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _blend(a: float, b: float, alpha: float) -> float:
    """
    alpha = how much to trust b
    """
    return (1.0 - alpha) * float(a) + alpha * float(b)


def _founder_contribution(proposal: Dict, state) -> Dict[str, float]:
    return {
        "price": float(proposal.get("price", state.product.price)),
        "quality": float(proposal.get("quality", state.product.quality)),
        "marketing_budget": float(state.marketing.budget),
    }


def _marketing_contribution(proposal: Dict, state) -> Dict[str, float]:
    return {
        "price": float(state.product.price),
        "quality": float(state.product.quality),
        "marketing_budget": float(proposal.get("marketing_budget", state.marketing.budget)),
    }


def _investor_contribution(proposal: Dict, state) -> Dict[str, float]:
    risk_level = str(proposal.get("risk_level", "medium")).lower()
    approve = proposal.get("approve", True)
    risk_scale = {"low": 1.1, "medium": 1.0, "high": 0.75}.get(risk_level, 1.0)
    if approve is False:
        risk_scale *= 0.6
    return {
        "price": float(state.product.price),
        "quality": float(state.product.quality),
        "marketing_budget": float(state.marketing.budget) * risk_scale,
    }


def _customer_contribution(proposal: Dict, state) -> Dict[str, float]:
    feedback = proposal.get("feedback_score")
    sentiment = str(proposal.get("sentiment", "neutral")).lower()
    quality = state.product.quality
    price = state.product.price

    if feedback is not None:
        gap = 0.5 - float(feedback)
        quality = quality + gap * 0.3
        price = price - gap * 200.0
    elif sentiment == "negative":
        quality += 0.05
        price -= 50.0
    elif sentiment == "positive":
        price += 25.0

    return {
        "price": float(price),
        "quality": float(quality),
        "marketing_budget": float(state.marketing.budget),
    }


_CONTRIBUTORS = {
    "founder": _founder_contribution,
    "marketing": _marketing_contribution,
    "investor": _investor_contribution,
    "customer": _customer_contribution,
}


def _current_state_decision(state) -> Dict[str, float]:
    return {
        "price": float(state.product.price),
        "quality": float(state.product.quality),
        "marketing_budget": float(state.marketing.budget),
    }


def fuse_decision(
    agent_outputs: List[Dict],
    weights: Dict[str, float],
    state,
    reference_decision: Optional[Dict] = None,
    ppo_value_estimate: Optional[float] = None,
) -> Dict[str, float]:
    """
    Weighted LLM fusion plus optional PPO anchor.

    If PPO is available, it acts as a stabilizing reference rather than a
    competing decision-maker. This helps the final decision stay grounded
    even when one or more agents fail.
    """
    current = _current_state_decision(state)

    if not weights:
        if reference_decision:
            alpha = 0.30  # use PPO more strongly if all agents failed
            return {
                "price": _clamp(_blend(current["price"], reference_decision.get("price", current["price"]), alpha), *PRICE_RANGE),
                "quality": _clamp(_blend(current["quality"], reference_decision.get("quality", current["quality"]), alpha), *QUALITY_RANGE),
                "marketing_budget": _clamp(_blend(current["marketing_budget"], reference_decision.get("marketing_budget", current["marketing_budget"]), alpha), *BUDGET_RANGE),
            }

        return current

    totals = {"price": 0.0, "quality": 0.0, "marketing_budget": 0.0}
    by_name = {o["agent"]: o for o in agent_outputs}

    for name, weight in weights.items():
        output = by_name.get(name)
        if output is None:
            continue

        proposal = output.get("proposal", {}) or {}
        contributor = _CONTRIBUTORS.get(name)
        if contributor is None:
            continue

        contribution = contributor(proposal, state)
        for field in totals:
            totals[field] += weight * contribution[field]

    final = totals

    # PPO anchor: stronger when fewer agents survived, weaker when all agents are active.
    if reference_decision:
        surviving_agents = max(1, len(weights))
        base_alpha = 0.18
        missing_agent_bonus = min(0.14, (4 - surviving_agents) * 0.04)

        # If the critic thinks the PPO value is strong, trust it slightly more.
        value_bonus = 0.0
        if ppo_value_estimate is not None:
            value_bonus = max(-0.03, min(0.08, float(ppo_value_estimate) / 500.0))

        alpha = _clamp(base_alpha + missing_agent_bonus + value_bonus, 0.12, 0.42)

        final = {
            "price": _blend(final["price"], reference_decision.get("price", final["price"]), alpha),
            "quality": _blend(final["quality"], reference_decision.get("quality", final["quality"]), alpha),
            "marketing_budget": _blend(final["marketing_budget"], reference_decision.get("marketing_budget", final["marketing_budget"]), alpha),
        }

    return {
        "price": _clamp(final["price"], *PRICE_RANGE),
        "quality": _clamp(final["quality"], *QUALITY_RANGE),
        "marketing_budget": _clamp(final["marketing_budget"], *BUDGET_RANGE),
    }