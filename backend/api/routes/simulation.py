from fastapi import APIRouter
from backend.services import simulation_service as service

router = APIRouter()


@router.get("/start")
def start_simulation(steps: int = 10):
    return service.start(steps)


@router.get("/step")
def step_simulation():
    return service.step()


@router.get("/reset")
def reset_simulation():
    return service.reset()


@router.get("/stop")
def stop_simulation():
    return service.stop()