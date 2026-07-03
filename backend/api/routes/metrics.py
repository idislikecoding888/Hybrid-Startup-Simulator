from fastapi import APIRouter
from backend.services.metrics_service import MetricsService

router = APIRouter()
metrics_service = MetricsService()


@router.get("/current")
def get_current_metrics():
    return metrics_service.get_current_metrics()


@router.get("/history")
def get_metrics_history():
    return metrics_service.get_metrics_history()


@router.get("/summary")
def get_summary():
    return metrics_service.get_latest_summary()