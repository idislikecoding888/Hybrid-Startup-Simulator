from typing import Dict, List, Optional
import math

PRICE_RANGE = (100.0, 2000.0)
QUALITY_RANGE = (0.30, 1.0)
MARKETING_RANGE = (0.0, 5000.0)


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_inventory_stock(state) -> float:
    inventory = getattr(state, "inventory", None)
    if isinstance(inventory, dict):
        return _safe_float(inventory.get("stock", 0), 0.0)

    stock = getattr(inventory, "stock", None)
    return _safe_float(stock, 0.0)


def _safe_cash(state) -> float:
    return _safe_float(getattr(getattr(state, "finance", None), "cash", 0.0), 0.0)


def _safe_satisfaction(state) -> float:
    return clamp(
        _safe_float(getattr(getattr(state, "customers", None), "satisfaction", 0.5), 0.5),
        0.0,
        1.0,
    )


def _ppo_value_signal(ppo_value_estimate) -> float:
    """
    Maps PPO value estimates into a stable 0..1 signal.
    Higher value = stronger confidence in the current regime.
    """
    if ppo_value_estimate is None:
        return 0.5
    try:
        v = float(ppo_value_estimate)
    except (TypeError, ValueError):
        return 0.5

    # Smoothly map to 0..1 without assuming a fixed PPO scale.
    return clamp(0.5 + 0.5 * math.tanh(v / 50.0), 0.0, 1.0)


def _business_risk(state) -> float:
    """
    Returns a 0..1 risk score:
    higher = more need for conservative / PPO-stabilized fusion.
    """
    cash = _safe_cash(state)
    stock = _safe_inventory_stock(state)
    satisfaction = _safe_satisfaction(state)

    # Cash risk: low cash -> high risk
    if cash <= 0:
        cash_risk = 1.0
    elif cash < 50000:
        cash_risk = 0.85
    elif cash > 250000:
        cash_risk = 0.10
    else:
        cash_risk = (250000 - cash) / 200000.0
        cash_risk = clamp(cash_risk, 0.10, 0.85)

    # Inventory risk: low stock -> high risk
    if stock <= 0:
        inventory_risk = 1.0
    elif stock < 50:
        inventory_risk = 0.90
    elif stock > 300:
        inventory_risk = 0.10
    else:
        inventory_risk = (300 - stock) / 250.0
        inventory_risk = clamp(inventory_risk, 0.10, 0.90)

    # Satisfaction risk: low satisfaction -> high risk
    satisfaction_risk = 1.0 - clamp(satisfaction, 0.0, 1.0)

    return clamp(
        0.40 * cash_risk + 0.25 * inventory_risk + 0.35 * satisfaction_risk,
        0.0,
        1.0,
    )


def founder_strategy(state, proposal):
    return {
        "price": proposal.get("price", state.product.price),
        "quality": proposal.get("quality", state.product.quality),
        "marketing_budget": state.marketing.budget,
    }


def marketing_strategy(state, proposal):
    return {
        "price": state.product.price,
        "quality": state.product.quality,
        "marketing_budget": proposal.get(
            "marketing_budget",
            state.marketing.budget,
        ),
    }


def investor_strategy(state, proposal):
    approve = proposal.get("approve", True)
    risk = proposal.get("risk_level", "medium")

    multiplier = {
        "low": 1.10,
        "medium": 1.0,
        "high": 0.75,
    }.get(risk, 1.0)

    if approve is False:
        multiplier *= 0.70

    return {
        "price": state.product.price,
        "quality": state.product.quality,
        "marketing_budget": state.marketing.budget * multiplier,
    }


def customer_strategy(state, proposal):
    quality = state.product.quality
    price = state.product.price

    sentiment = proposal.get("sentiment", "neutral")
    feedback = proposal.get("feedback_score", 0.5)

    if sentiment == "negative":
        quality += 0.05
        price -= 40
    elif sentiment == "positive":
        price += 25

    if feedback < 0.5:
        quality += (0.5 - feedback) * 0.25

    return {
        "price": price,
        "quality": quality,
        "marketing_budget": state.marketing.budget,
    }


STRATEGIES = {
    "founder": founder_strategy,
    "marketing": marketing_strategy,
    "investor": investor_strategy,
    "customer": customer_strategy,
}


def proposal_utility(agent_name, proposal, state, weight):
    utility = max(0.0, float(weight))

    confidence = proposal.get("confidence", 0.5)
    confidence = clamp(_safe_float(confidence, 0.5), 0.0, 1.0)
    utility *= 0.55 + 0.45 * confidence

    if agent_name == "marketing":
        if state.finance.cash > 100000:
            utility *= 1.10

        inventory_stock = _safe_inventory_stock(state)

        if inventory_stock < 100:
            utility *= 0.75

    elif agent_name == "founder":
        if state.customers.satisfaction < 0.70:
            utility *= 1.15

    elif agent_name == "customer":
        fb = proposal.get("feedback_score", 0.5)
        fb = clamp(_safe_float(fb, 0.5), 0.0, 1.0)
        utility *= (0.5 + fb)

    elif agent_name == "investor":
        if proposal.get("approve", True):
            utility *= 1.05
        else:
            utility *= 0.90

    return utility


def _weighted_consensus(candidates: List[Dict]) -> Dict[str, float]:
    """
    Utility-weighted consensus across the surviving candidate decisions.
    """
    if not candidates:
        return {}

    total = sum(max(0.0, float(c["utility"])) for c in candidates)
    if total <= 0:
        return candidates[0]["decision"]

    consensus = {
        "price": 0.0,
        "quality": 0.0,
        "marketing_budget": 0.0,
    }

    for c in candidates:
        w = max(0.0, float(c["utility"])) / total
        d = c["decision"]
        consensus["price"] += w * float(d["price"])
        consensus["quality"] += w * float(d["quality"])
        consensus["marketing_budget"] += w * float(d["marketing_budget"])

    return consensus


def _blend_decisions(base: Dict[str, float], target: Dict[str, float], alpha: float) -> Dict[str, float]:
    """
    alpha = how much to trust the target decision.
    """
    alpha = clamp(alpha, 0.0, 1.0)
    return {
        "price": (1.0 - alpha) * float(base["price"]) + alpha * float(target["price"]),
        "quality": (1.0 - alpha) * float(base["quality"]) + alpha * float(target["quality"]),
        "marketing_budget": (1.0 - alpha) * float(base["marketing_budget"]) + alpha * float(target["marketing_budget"]),
    }


def fuse_decision(
    agent_outputs,
    weights,
    state,
    reference_decision=None,
    ppo_value_estimate=None,
    agent_stats=None,
):
    candidates = []

    for output in agent_outputs:
        agent = output["agent"]

        if agent not in STRATEGIES:
            continue

        if output.get("error"):
            continue

        proposal = output.get("proposal", {}) or {}

        candidate = STRATEGIES[agent](
            state,
            proposal,
        )

        utility = proposal_utility(
            agent,
            proposal,
            state,
            weights.get(agent, 0),
        )

        stats = (agent_stats or {}).get(agent, {})

        historical_success = stats.get("successful_decisions", 0)
        historical_average = stats.get("average_reward", 0.5)

        history_factor = (
            0.75
            + 0.10 * min(historical_success / 10.0, 1.0)
            + 0.15 * historical_average
        )

        utility *= history_factor

        candidates.append({
            "agent": agent,
            "decision": candidate,
            "utility": utility,
            "reasoning": output.get("reasoning", ""),
        })

        ppo_alignment = 1.0

        if reference_decision:

            decision = STRATEGIES[agent](state, proposal)

            price_diff = abs(
                decision["price"] -
                reference_decision["price"]
            ) / PRICE_RANGE[1]

            quality_diff = abs(
                decision["quality"] -
                reference_decision["quality"]
            )

            marketing_diff = abs(
                decision["marketing_budget"] -
                reference_decision["marketing_budget"]
            ) / MARKETING_RANGE[1]

            disagreement = (
                price_diff +
                quality_diff +
                marketing_diff
            ) / 3.0

            ppo_alignment = max(
                0.60,
                1.0 - disagreement
            )

        utility *= ppo_alignment

    if not candidates:
        return {
            "price": state.product.price,
            "quality": state.product.quality,
            "marketing_budget": state.marketing.budget,
        }

    candidates.sort(key=lambda x: x["utility"], reverse=True)
    winner = candidates[0]
    winner_decision = winner["decision"]

    # If the top candidates are close, let the consensus matter more.
    top_utility = float(candidates[0]["utility"])
    second_utility = float(candidates[1]["utility"]) if len(candidates) > 1 else 0.0
    dominance_gap = clamp(top_utility - second_utility, 0.0, 1.0)

    consensus = _weighted_consensus(candidates)

    business_risk = _business_risk(state)
    value_signal = _ppo_value_signal(ppo_value_estimate)

    # Sprint 3 behavior:
    # - stronger consensus blend when the board is split
    # - stronger PPO anchor when business risk is high or PPO value is weak
    consensus_alpha = clamp(
        0.12 + 0.35 * (1.0 - dominance_gap) + 0.10 * business_risk,
        0.10,
        0.60,
    )

    decision = _blend_decisions(winner_decision, consensus, consensus_alpha)

    if reference_decision and getattr(__import__("backend.config", fromlist=["settings"]).settings, "HYBRID_PPO_ENABLED", True):
        ppo_alpha = clamp(
            0.10 + 0.18 * business_risk + 0.10 * (1.0 - value_signal) + 0.06 * (1.0 - top_utility),
            0.08,
            0.40,
        )
        decision = _blend_decisions(decision, reference_decision, ppo_alpha)

    return {
        "price": clamp(decision["price"], *PRICE_RANGE),
        "quality": clamp(decision["quality"], *QUALITY_RANGE),
        "marketing_budget": clamp(decision["marketing_budget"], *MARKETING_RANGE),
    }