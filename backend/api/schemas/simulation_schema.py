from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class StartSimulationRequest(BaseModel):
    """Request to start the simulation."""
    initial_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional initial simulation parameters."
    )


class StopSimulationRequest(BaseModel):
    """Request to stop the simulation."""
    force: Optional[bool] = Field(
        default=False,
        description="If true, stop immediately even if cleanup is required."
    )


class StepSimulationRequest(BaseModel):
    """Request to advance the simulation by one or more steps."""
    steps: int = Field(
        default=1,
        ge=1,
        description="Number of simulation steps to execute."
    )


class SimulationCommandResponse(BaseModel):
    """Generic response for simulation control endpoints."""
    status: str = Field(..., description="The result status of the command.")
    message: str = Field(..., description="A human-readable response message.")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional command-specific details or payload."
    )
