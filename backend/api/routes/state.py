from fastapi import APIRouter
from backend.services import simulation_service as service

router = APIRouter()


@router.get("/")
def get_state():
    return service.get_state()