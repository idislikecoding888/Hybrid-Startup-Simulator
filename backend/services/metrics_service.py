# backend/services/metrics_service.py

from backend.services import simulation_service
from backend.metrics.metrics_engine import compute_metrics


class MetricsService:
    def __init__(self):
        self.simulation_service = simulation_service

    def get_current_metrics(self) -> dict:
        """
        Compute metrics from current simulation state
        """
        state = self.simulation_service.engine.state
        return compute_metrics(state)

    def get_metrics_history(self) -> list:
        """
        Return metrics history from state history
        """
        state = self.simulation_service.engine.state

        history = []
        for entry in state.history:
            history.append({
                "step": entry.get("step"),
                "metrics": entry.get("metrics")
            })

        return history

    def get_latest_summary(self) -> dict:
        """
        Return latest snapshot summary
        """
        state = self.simulation_service.engine.state

        return {
            "step": state.step,
            "cash": state.finance.cash,
            "revenue": state.finance.revenue,
            "customers": state.customers.total_customers,
            "satisfaction": state.customers.satisfaction
        }