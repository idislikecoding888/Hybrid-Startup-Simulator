from typing import Dict


class ConversionCalculator:
    """
    Calculates conversion rate based on:
    - product quality
    - pricing attractiveness
    - marketing effectiveness
    """

    def __init__(self):
        pass

    def calculate(self, state: Dict) -> float:
        """
        Conversion rate model:

        conversion_rate =
            base_rate
            + quality_boost
            + marketing_boost
            - price_penalty
        """

        price = state["product"]["price"]
        quality = state["product"]["quality"]
        marketing_budget = state["marketing"]["budget"]

        base_rate = 0.02  # baseline conversion

        # Quality improves trust → more conversions
        quality_boost = 0.1 * quality

        # Marketing increases awareness
        marketing_boost = 0.00001 * marketing_budget

        # Higher price reduces conversion
        price_penalty = 0.00005 * price

        conversion_rate = (
            base_rate
            + quality_boost
            + marketing_boost
            - price_penalty
        )

        # Clamp between 0 and 1
        conversion_rate = max(0.0, min(1.0, conversion_rate))

        return conversion_rate

    def calculate_detailed(self, state: Dict) -> Dict:
        """
        Returns breakdown for debugging / UI
        """

        price = state["product"]["price"]
        quality = state["product"]["quality"]
        marketing_budget = state["marketing"]["budget"]

        base_rate = 0.02
        quality_boost = 0.1 * quality
        marketing_boost = 0.00001 * marketing_budget
        price_penalty = 0.00005 * price

        conversion_rate = (
            base_rate
            + quality_boost
            + marketing_boost
            - price_penalty
        )

        conversion_rate = max(0.0, min(1.0, conversion_rate))

        return {
            "base_rate": base_rate,
            "quality_boost": quality_boost,
            "marketing_boost": marketing_boost,
            "price_penalty": price_penalty,
            "final_conversion_rate": conversion_rate
        }