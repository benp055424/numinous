from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

VERICORE_COST_PER_CALL = Decimal("0.05")


class VericoreStatement(BaseModel):
    statement: str
    url: str
    contradiction: float = Field(..., ge=0, le=1)
    neutral: float = Field(..., ge=0, le=1)
    entailment: float = Field(..., ge=0, le=1)
    sentiment: float
    conviction: float = Field(..., ge=0, le=1)
    source_credibility: float = Field(..., ge=0, le=1)
    narrative_momentum: float = Field(..., ge=0, le=1)
    risk_reward_sentiment: float
    political_leaning: float = 0.0
    catalyst_detection: float = Field(..., ge=0, le=1)

    model_config = ConfigDict(extra="allow")


class VericoreEvidenceSummary(BaseModel):
    total_count: int
    support: Optional[float] = None
    neutral: float
    refute: Optional[float] = None
    entailment: Optional[float] = None
    contradiction: Optional[float] = None
    sentiment: float
    conviction: float
    source_credibility: float
    narrative_momentum: float
    risk_reward_sentiment: float
    political_leaning: float = 0.0
    catalyst_detection: float
    statements: list[VericoreStatement]

    model_config = ConfigDict(extra="allow")


class VericoreResponse(BaseModel):
    batch_id: str
    request_id: str
    preview_url: str
    evidence_summary: VericoreEvidenceSummary

    model_config = ConfigDict(extra="allow")


def calculate_cost() -> Decimal:
    return VERICORE_COST_PER_CALL
