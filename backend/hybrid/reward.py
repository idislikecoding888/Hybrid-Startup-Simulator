# backend/hybrid/reward.py
#
# Multi-objective reward for the PPO weighting layer. Combines several
# business objectives (rather than optimizing a single metric) so PPO
# learns to weight LLM agents in a way that sustains the business, not
# just maximize one number.
#
# This reward is computed every step for logging/analysis now, and is
# exactly what a future training loop (see ppo_adapter.py docstring)
# would feed into PPO's rollout buffer to fine-tune the weighting head.

from typing import Dict

# Relative importance of each objective. Kept explicit + tunable in one
# place rather than baked into the formula.
REWARD_WEIGHTS = {
    "revenue_growth": 0.20,
    "profit": 0.20,
    "customer_satisfaction": 0.15,
    "retention": 0.15,
    "market_share_proxy": 0.10,
    "survival": 0.10,
    "investment_success": 0.05,
    "product_quality": 0.05,
}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_reward(prev_state, new_state, metrics: Dict) -> Dict:
    """
    prev_state / new_state: SimulationState before/after the step.
    metrics: output of metrics/metrics_engine.compute_metrics(new_state).

    Returns {"total": float, "components": {...}} so both the scalar
    reward and its breakdown can be logged for research analysis.
    """
    revenue_growth = _safe_div(
        new_state.finance.revenue - prev_state.finance.revenue,
        max(prev_state.finance.revenue, 1.0),
    )
    revenue_growth = max(-1.0, min(1.0, revenue_growth))

    profit = new_state.finance.revenue - new_state.finance.expenses
    profit_norm = max(-1.0, min(1.0, _safe_div(profit, max(new_state.finance.expenses, 1.0))))

    customer_satisfaction = new_state.customers.satisfaction  # already 0..1

    retention = metrics.get("retention", 0.0)

    # Market share proxy: growth in active customer base.
    market_share_proxy = max(
        -1.0,
        min(
            1.0,
            _safe_div(
                new_state.customers.active_customers - prev_state.customers.active_customers,
                max(prev_state.customers.active_customers, 1.0),
            ),
        ),
    )

    # Startup survival: still solvent.
    survival = 1.0 if new_state.finance.cash > 0 else 0.0

    # Investment success proxy: cash is growing (investor's capital is
    # being protected/grown, not burned).
    investment_success = 1.0 if new_state.finance.cash >= prev_state.finance.cash else 0.0

    product_quality = new_state.product.quality  # already 0..1

    components = {
        "revenue_growth": revenue_growth,
        "profit": profit_norm,
        "customer_satisfaction": customer_satisfaction,
        "retention": retention,
        "market_share_proxy": market_share_proxy,
        "survival": survival,
        "investment_success": investment_success,
        "product_quality": product_quality,
    }

    total = sum(REWARD_WEIGHTS[k] * components[k] for k in REWARD_WEIGHTS)

    return {"total": float(total), "components": components}
