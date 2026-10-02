from datetime import date
from typing import Optional
from pydantic import BaseModel, Field


class OrderInput(BaseModel):
    item_price: float = Field(gt=0)
    item_size: str
    item_color: str
    brand_id: int
    user_title: str
    user_state: str
    order_date: date
    user_reg_date: date
    user_dob: Optional[date] = None        # optional
    delivery_date: Optional[date] = None   # optional


class PredictionResponse(BaseModel):
    return_probability: float
    no_return_probability: float
    prediction: int
    label: str