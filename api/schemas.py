from typing import List
from pydantic import BaseModel, Field


class Reading(BaseModel):
    cycle: int
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class RULRequest(BaseModel):
    readings: List[Reading] = Field(..., min_length=1)


class RULResponse(BaseModel):
    predicted_rul: float
    cycles_used: int