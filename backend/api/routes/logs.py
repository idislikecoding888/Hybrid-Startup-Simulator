from fastapi import APIRouter
from backend.services.deliberation_service import DeliberationService

router = APIRouter()
from backend.services import simulation_service

deliberation_service = simulation_service.deliberation_service


@router.get("/")
def get_logs():
    logs = deliberation_service.get_logs()
    return {"logs": logs}


@router.get("/latest")
def get_latest_log():
    log = deliberation_service.get_latest_log()
    return log