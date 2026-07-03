from typing import Dict


class RevenueCalculator:
    """
    Calculates revenue based on:
    - product price
    - number of converted customers
    """

    def __init__(self):
        pass

    def calculate(self, state: Dict) -> float:
        """
        Revenue = price * number of purchases

        Purchases are derived from:
        active_customers * conversion_rate
        """

        price = state["product"]["price"]
        active_customers = state["customers"]["active_customers"]
        conversion_rate = state["marketing"]["conversion_rate"]

        purchases = int(active_customers * conversion_rate)

        revenue = price * purchases

        return revenue

    def calculate_detailed(self, state: Dict) -> Dict:
        """
        Returns detailed breakdown (useful for logs/UI/debugging)
        """

        price = state["product"]["price"]
        active_customers = state["customers"]["active_customers"]
        conversion_rate = state["marketing"]["conversion_rate"]

        purchases = int(active_customers * conversion_rate)
        revenue = price * purchases

        return {
            "price": price,
            "active_customers": active_customers,
            "conversion_rate": conversion_rate,
            "purchases": purchases,
            "revenue": revenue
        }