# backend/hybrid/fusion.py
#
# Combines every LLM agent's proposal into ONE final business decision
# using the PPO-derived weights:
#
#   final[field] = sum_i( weight_i * agent_i_contribution[field] )
#
# Every agent gets an opinion on every decision field (price, quality,
# marketing_budget), even fields outside its usual domain, so the weighted
# sum in the spec ("Founder x w1 + Marketing x w2 + Investor x w3 +
# Customer x w4") is literal rather than only applying to whichever single
# agent happens to own that field. Agents with no direct opinion on a
# field contribute a "hold current value" vote, which naturally has low
# leverage once combined with agents that do have an opinion.

from typing import Dict, List

PRICE_RANGE = (100.0, 2000.0)
QUALITY_RANGE = (0.1, 1.0)
BUDGET_RANGE = (100.0, 5000.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _founder_contribution(proposal: Dict, state) -> Dict[str, float]:
    return {
        "price": float(proposal.get("price", state.product.price)),
        "quality": float(proposal.get("quality", state.product.quality)),
        "marketing_budget": float(state.marketing.budget),  # no opinion -> hold
    }


def _marketing_contribution(proposal: Dict, state) -> Dict[str, float]:
    return {
        "price": float(state.product.price),  # no opinion -> hold
        "quality": float(state.product.quality),  # no opinion -> hold
        "marketing_budget": float(proposal.get("marketing_budget", state.marketing.budget)),
    }


def _investor_contribution(proposal: Dict, state) -> Dict[str, float]:
    risk_level = str(proposal.get("risk_level", "medium")).lower()
    approve = proposal.get("approve", True)
    risk_scale = {"low": 1.1, "medium": 1.0, "high": 0.75}.get(risk_level, 1.0)
    if approve is False:
        risk_scale *= 0.6  # veto -> pull spend/price down
    return {
        "price": float(state.product.price),  # no direct price opinion -> hold
        "quality": float(state.product.quality),  # no direct quality opinion -> hold
        "marketing_budget": float(state.marketing.budget) * risk_scale,
    }


def _customer_contribution(proposal: Dict, state) -> Dict[str, float]:
    feedback = proposal.get("feedback_score")
    sentiment = str(proposal.get("sentiment", "neutral")).lower()
    quality = state.product.quality
    price = state.product.price
    if feedback is not None:
        # Low feedback -> customers want more quality for the price / a
        # lower price; high feedback -> current trade-off is fine.
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
        "marketing_budget": float(state.marketing.budget),  # no opinion -> hold
    }


_CONTRIBUTORS = {
    "founder": _founder_contribution,
    "marketing": _marketing_contribution,
    "investor": _investor_contribution,
    "customer": _customer_contribution,
}


def fuse_decision(
    agent_outputs: List[Dict],
    weights: Dict[str, float],
    state,
) -> Dict[str, float]:
    """
    agent_outputs: list of {"agent": name, "proposal": {...}, ...}
    weights: {agent_name: weight}, from weighting.compute_agent_weights
    Returns clamped {price, quality, marketing_budget} final decision.

    Robustness: agents missing from `weights` (failed / excluded upstream)
    are simply skipped here too -- their weight has already been
    redistributed among the rest by compute_agent_weights.
    """
    if not weights:
        # Total fusion failure fallback: hold current state.
        return {
            "price": float(state.product.price),
            "quality": float(state.product.quality),
            "marketing_budget": float(state.marketing.budget),
        }

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

    return {
        "price": _clamp(totals["price"], *PRICE_RANGE),
        "quality": _clamp(totals["quality"], *QUALITY_RANGE),
        "marketing_budget": _clamp(totals["marketing_budget"], *BUDGET_RANGE),
    }
