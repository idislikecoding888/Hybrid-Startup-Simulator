from typing import Dict, List, Optional

PRICE_RANGE = (100.0, 2000.0)
QUALITY_RANGE = (0.30, 1.0)
MARKETING_RANGE = (0.0, 5000.0)


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


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

    if feedback < .5:
        quality += (.5 - feedback) * .25

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


def proposal_utility(agent_name,
                     proposal,
                     state,
                     weight):

    utility = weight

    confidence = proposal.get("confidence", .5)

    utility *= confidence

    if agent_name == "marketing":

        if state.finance.cash > 100000:
            utility *= 1.10

        inventory_stock = getattr(state.inventory, "stock", None)

        if inventory_stock is None:
            try:
                inventory_stock = state.inventory.get("stock", 0)
            except Exception:
                inventory_stock = 0

        if inventory_stock < 100:
            utility *= 0.75

    elif agent_name == "founder":

        if state.customers.satisfaction < .70:
            utility *= 1.15

    elif agent_name == "customer":

        fb = proposal.get("feedback_score", .5)

        utility *= (0.5 + fb)

    elif agent_name == "investor":

        if proposal.get("approve", True):
            utility *= 1.05
        else:
            utility *= .90

    return utility


def fuse_decision(
    agent_outputs,
    weights,
    state,
    reference_decision=None,
    ppo_value_estimate=None,
):

    candidates = []

    for output in agent_outputs:

        agent = output["agent"]

        if agent not in STRATEGIES:
            continue

        if output.get("error"):
            continue

        proposal = output.get("proposal", {})

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

        candidates.append({
            "agent": agent,
            "decision": candidate,
            "utility": utility,
            "reasoning": output.get("reasoning", "")
        })

    if not candidates:

        return {
            "price": state.product.price,
            "quality": state.product.quality,
            "marketing_budget": state.marketing.budget,
        }

    candidates.sort(
        key=lambda x: x["utility"],
        reverse=True,
    )

    winner = candidates[0]

    decision = winner["decision"]

    if reference_decision:

        alpha = .20

        decision = {

            "price":
                (1-alpha)*decision["price"]
                +
                alpha*reference_decision["price"],

            "quality":
                (1-alpha)*decision["quality"]
                +
                alpha*reference_decision["quality"],

            "marketing_budget":
                (1-alpha)*decision["marketing_budget"]
                +
                alpha*reference_decision["marketing_budget"],
        }

    return {

        "price":
            clamp(
                decision["price"],
                *PRICE_RANGE
            ),

        "quality":
            clamp(
                decision["quality"],
                *QUALITY_RANGE
            ),

        "marketing_budget":
            clamp(
                decision["marketing_budget"],
                *MARKETING_RANGE
            ),
    }