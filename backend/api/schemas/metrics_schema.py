from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class KPIData(BaseModel):
    revenue: float = Field(..., description="Total revenue for the measurement period.")
    customers: int = Field(..., description="Number of customers acquired or active.")
    cac: float = Field(..., description="Customer acquisition cost.")
    conversion: float = Field(..., ge=0.0, le=1.0, description="Conversion rate as a fraction between 0 and 1.")


class MetricsResponse(BaseModel):
    timestamp: datetime = Field(..., description="Timestamp when the metrics were recorded.")
    kpis: KPIData = Field(..., description="Key performance indicators for the current metrics snapshot.")
    note: Optional[str] = Field(default=None, description="Optional note or context for the metrics.")


class MetricsHistoryResponse(BaseModel):
    history: List[MetricsResponse] = Field(..., description="Historical KPI metrics snapshots.")
