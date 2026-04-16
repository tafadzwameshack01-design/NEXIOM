"""
NEXOM — Pydantic v2 Schemas
All inter-service data structures per spec Section PYDANTIC v2 SCHEMAS
"""
from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GameStateSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    score_home: int = Field(ge=0)
    score_away: int = Field(ge=0)
    possession_team: Literal["home", "away"]
    time_remaining: float = Field(ge=0.0, le=720.0)
    period: int = Field(ge=1, le=5)
    lineup_home: list[int] = Field(min_length=5, max_length=5)
    lineup_away: list[int] = Field(min_length=5, max_length=5)
    fatigue_index_home: float = Field(ge=0.0, le=1.0)
    fatigue_index_away: float = Field(ge=0.0, le=1.0)
    pace_state: float = Field(ge=0.7, le=1.3)
    market_state: Literal["PRE_GAME", "LIVE_Q1", "LIVE_Q2", "LIVE_Q3", "LIVE_Q4", "LIVE_OT"]
    game_id: str = ""


class SimulationInputSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    lineup_home: list[int] = Field(min_length=5, max_length=5)
    lineup_away: list[int] = Field(min_length=5, max_length=5)
    pace: float = Field(ge=80.0, le=115.0)
    N: int

    @field_validator("N")
    @classmethod
    def validate_n(cls, v: int) -> int:
        allowed = {1000, 5000, 10000, 25000, 100000}
        if v not in allowed:
            raise ValueError(f"N must be one of {allowed}")
        return v


class SimulationOutputSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    p_home_win: float = Field(ge=0.001, le=0.999)
    p_cover_spread: float = Field(ge=0.001, le=0.999)
    p_over_total: float = Field(ge=0.001, le=0.999)
    p_overtime: float = Field(ge=0.001, le=0.999)
    p_home_cover_alt: float = Field(ge=0.001, le=0.999)
    game_id: str = ""
    n_simulations: int = 0
    elapsed_seconds: float = 0.0


class BetSignalSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    p_final: float = Field(ge=0.0, le=1.0)
    p_market: float = Field(ge=0.0, le=1.0)
    ev_exec: float
    kelly_fraction: float = Field(ge=0.0, le=0.35)
    stake: float = Field(gt=0.0)
    bookmaker: str
    market_type: Literal["moneyline", "spread", "total", "player_prop"]
    game_id: str
    all_gates_passed: bool
    failed_gates: list[str]


class PlacedBetSchema(BetSignalSchema):
    bet_id: UUID = Field(default_factory=uuid4)
    placed_at: datetime = Field(default_factory=datetime.utcnow)
    odds_at_placement: float
    fill_confirmed: bool = False
    fill_timestamp: Optional[datetime] = None


class ResolvedBetSchema(PlacedBetSchema):
    outcome: bool
    pnl: float
    clv: float
    brier_contribution: float


class DCParameterSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    team_id: int
    alpha: float = Field(ge=0.5, le=2.0)
    beta: float = Field(ge=0.5, le=2.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    game_count_in_window: int = 0


class RiskEventSchema(BaseModel):
    model_config = ConfigDict(strict=True)

    event_type: Literal[
        "STOP_7D_DRAWDOWN",
        "STOP_DAILY_LOSS",
        "STOP_CATEGORY_STREAK",
        "STOP_VOLATILITY_SPIKE",
        "RUIN_GUARD",
        "DRIFT_DETECTED",
        "CALIBRATION_FAILURE",
        "REGIME_CHANGE",
    ]
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    details: dict
