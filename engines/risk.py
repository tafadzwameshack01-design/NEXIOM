"""
NEXOM — Risk Management System
Section J of the NEXOM v5 specification.

Implements:
  J.1  Hard stop conditions
  J.2  Ruin probability guard
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np


StopType = Literal[
    "STOP_7D_DRAWDOWN",
    "STOP_DAILY_LOSS",
    "STOP_CATEGORY_STREAK",
    "STOP_VOLATILITY_SPIKE",
    "RUIN_GUARD",
]

MarketCategory = Literal["moneyline", "spread", "total", "player_prop"]


@dataclass
class ActiveStop:
    stop_type: StopType
    triggered_at: datetime
    expires_at: datetime
    category: MarketCategory | None = None   # for STOP_CATEGORY_STREAK
    details: dict = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return datetime.utcnow() < self.expires_at

    @property
    def remaining_hours(self) -> float:
        delta = self.expires_at - datetime.utcnow()
        return max(0.0, delta.total_seconds() / 3600.0)


@dataclass
class RiskState:
    """Full risk state for the system."""
    bankroll: float
    bankroll_7d_ago: float
    bankroll_30d_ago: float
    daily_pnl: float = 0.0
    active_stops: list[ActiveStop] = field(default_factory=list)
    kelly_fraction_multiplier: float = 0.30
    pnl_history_90d: list[float] = field(default_factory=list)  # daily P&L values
    win_count_30d: int = 15
    loss_count_30d: int = 10
    category_streaks: dict[str, int] = field(default_factory=dict)   # category → consecutive losses

    @property
    def bankroll_at_risk(self) -> float:
        return max(self.bankroll_30d_ago, self.bankroll)

    @property
    def drawdown_7d(self) -> float:
        if self.bankroll_at_risk <= 0:
            return 0.0
        return (self.bankroll_at_risk - self.bankroll) / self.bankroll_at_risk

    @property
    def daily_loss_fraction(self) -> float:
        if self.bankroll <= 0:
            return 0.0
        return -self.daily_pnl / self.bankroll if self.daily_pnl < 0 else 0.0

    def is_betting_suspended(self, category: MarketCategory | None = None) -> tuple[bool, list[str]]:
        """Returns (suspended, list_of_active_stop_names)."""
        self.active_stops = [s for s in self.active_stops if s.is_active]
        blocking = []
        for stop in self.active_stops:
            if stop.stop_type == "STOP_CATEGORY_STREAK":
                if category is None or stop.category == category:
                    blocking.append(stop.stop_type)
            else:
                blocking.append(stop.stop_type)
        return len(blocking) > 0, blocking


def check_7d_drawdown(state: RiskState) -> ActiveStop | None:
    """J.1 — STOP_7D_DRAWDOWN: 7-day rolling drawdown > 12% of bankroll-at-risk."""
    if state.drawdown_7d > 0.12:
        return ActiveStop(
            stop_type="STOP_7D_DRAWDOWN",
            triggered_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=48),
            details={"drawdown_pct": state.drawdown_7d * 100, "bankroll_at_risk": state.bankroll_at_risk},
        )
    return None


def check_daily_loss(state: RiskState) -> ActiveStop | None:
    """J.1 — STOP_DAILY_LOSS: daily P&L < −4% of bankroll."""
    if state.daily_loss_fraction > 0.04:
        # Halt until end of UTC day
        now = datetime.utcnow()
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return ActiveStop(
            stop_type="STOP_DAILY_LOSS",
            triggered_at=now,
            expires_at=midnight,
            details={"daily_loss_pct": state.daily_loss_fraction * 100},
        )
    return None


def check_category_streak(
    state: RiskState,
    category: MarketCategory,
    consecutive_losses: int,
) -> ActiveStop | None:
    """J.1 — STOP_CATEGORY_STREAK: 7 consecutive losses in a market category."""
    if consecutive_losses >= 7:
        return ActiveStop(
            stop_type="STOP_CATEGORY_STREAK",
            triggered_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=48),
            category=category,
            details={"category": category, "consecutive_losses": consecutive_losses},
        )
    return None


def check_volatility_spike(state: RiskState) -> ActiveStop | None:
    """J.1 — STOP_VOLATILITY_SPIKE: 7-day P&L std > 3× 90-day baseline std."""
    if len(state.pnl_history_90d) < 7:
        return None
    arr = np.array(state.pnl_history_90d)
    std_90d = float(np.std(arr))
    std_7d = float(np.std(arr[-7:]))
    if std_90d > 0 and std_7d > 3.0 * std_90d:
        return ActiveStop(
            stop_type="STOP_VOLATILITY_SPIKE",
            triggered_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=72),
            details={"std_7d": std_7d, "std_90d": std_90d, "ratio": std_7d / std_90d},
        )
    return None


def compute_ruin_probability(
    bankroll: float,
    avg_stake: float,
    win_rate_30d: float,
) -> float:
    """
    J.2 — Gambler's Ruin approximation.
    P(ruin) ≈ ((1−p_edge) / p_edge)^(bankroll / avg_stake)
    p_edge = fraction of placed bets historically winning (30-day).
    """
    p = float(np.clip(win_rate_30d, 0.01, 0.99))
    if avg_stake <= 0 or bankroll <= 0:
        return 0.0
    ratio = (1.0 - p) / p
    exponent = bankroll / avg_stake
    # Avoid overflow
    if ratio >= 1.0:
        return 1.0
    try:
        p_ruin = ratio ** exponent
    except OverflowError:
        p_ruin = 1.0
    return float(np.clip(p_ruin, 0.0, 1.0))


def ruin_guard(
    state: RiskState,
    avg_stake: float,
) -> tuple[float, bool]:
    """
    J.2 — If P(ruin) > 0.002, reduce kelly_fraction_multiplier by 0.05.
    Returns (new_kelly_multiplier, guard_activated).
    """
    total_bets = state.win_count_30d + state.loss_count_30d
    win_rate = state.win_count_30d / max(total_bets, 1)
    p_ruin = compute_ruin_probability(state.bankroll, avg_stake, win_rate)
    if p_ruin > 0.002:
        new_mult = max(0.0, state.kelly_fraction_multiplier - 0.05)
        return new_mult, True
    return state.kelly_fraction_multiplier, False


def run_all_risk_checks(
    state: RiskState,
    category: MarketCategory | None = None,
    avg_stake: float = 100.0,
) -> tuple[RiskState, list[str]]:
    """
    Run all risk checks and update state with any new stops.
    Returns (updated_state, list_of_triggered_stops).
    """
    triggered = []

    # Remove expired stops
    state.active_stops = [s for s in state.active_stops if s.is_active]
    existing_types = {s.stop_type for s in state.active_stops}

    checks = [
        ("STOP_7D_DRAWDOWN", check_7d_drawdown(state)),
        ("STOP_DAILY_LOSS", check_daily_loss(state)),
        ("STOP_VOLATILITY_SPIKE", check_volatility_spike(state)),
    ]
    if category:
        streak = state.category_streaks.get(category, 0)
        checks.append(("STOP_CATEGORY_STREAK", check_category_streak(state, category, streak)))

    for stop_name, stop in checks:
        if stop is not None and stop_name not in existing_types:
            state.active_stops.append(stop)
            triggered.append(stop_name)

    # Ruin guard
    new_mult, guard_fired = ruin_guard(state, avg_stake)
    if guard_fired and new_mult != state.kelly_fraction_multiplier:
        state.kelly_fraction_multiplier = new_mult
        triggered.append("RUIN_GUARD")

    # Volatility spike: reduce Kelly by 50% for 72h
    if "STOP_VOLATILITY_SPIKE" in triggered:
        state.kelly_fraction_multiplier *= 0.5

    return state, triggered
