# backend/environment/marketing_model.py

def compute_reach(marketing_budget: float) -> int:
    """
    Convert marketing budget to audience reach
    """
    return int(marketing_budget * 2)  # simple linear scaling


def compute_conversion_rate(quality: float) -> float:
    """
    Conversion rate influenced by product quality
    """

    base_rate = 0.05

    # quality impact
    if quality is None:
        quality = 0.5  # default baseline

    adjustment = (quality - 0.5) * 0.1

    conversion_rate = base_rate + adjustment

    # clamp between bounds
    return max(0.01, min(0.2, conversion_rate))


def apply_marketing(state, marketing_budget: float):
    """
    Apply marketing effects to state
    """

    reach = compute_reach(marketing_budget)
    conversion_rate = compute_conversion_rate(state.product.quality)

    # update state
    state.marketing.budget = marketing_budget
    state.marketing.reach = reach
    state.marketing.conversion_rate = conversion_rate

    return state, reach, conversion_rate