# backend/metrics/cac.py
def compute_cac(marketing_budget: float, new_customers: int) -> float:
    if new_customers <= 0:
        return float("inf")  # signals inefficiency

    return round(marketing_budget / new_customers, 2)