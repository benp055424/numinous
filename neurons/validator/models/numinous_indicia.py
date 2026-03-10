from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict


class IndiciaSignal(BaseModel):
    topic: str
    category: str
    signal: str
    confidence: str
    fact_status: str
    timestamp: datetime
    source_url: Optional[str] = None
    evidence_refs: list[str] = []

    model_config = ConfigDict(extra="allow")


class IndiciaSignalsResponse(BaseModel):
    signals: list[IndiciaSignal]

    model_config = ConfigDict(extra="allow")


def calculate_cost() -> Decimal:
    return Decimal("0")
