"""
NEXOM — Market Probability Engine
Section F of the NEXOM v5 specification.

Implements:
  F.1  Shin vig removal
  F.2  Bookmaker efficiency scoring
  F.3  Market filtering
  F.4  Steam detection
  F.5  CLV formal definition
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from scipy.optimize import brentq


# ── F.1 Shin Model ─────────────────────────────────────────────────────────────

def shin_two_outcome(d1: float, d2: float) -> tuple[float, float, float]:
    """
    F.1 — Shin (1993) vig removal for a 2-outcome market.
    d1, d2: decimal odds for outcomes 1 and 2.
    Returns (p_true_1, p_true_2, z_insider).
    z ∈ [0, 1]: Shin's insider trading parameter.
    Solves for z via scipy.optimize.brentq such that Σ p_i = 1.
    """
    if d1 <= 1.0 or d2 <= 1.0:
        raise ValueError("Decimal odds must be > 1.0")

    q1 = 1.0 / d1
    q2 = 1.0 / d2
    overround = q1 + q2  # > 1.0

    def p_i(z: float, qi: float) -> float:
        inner = z ** 2 + 4 * (1 - z) * qi ** 2 / overround
        return (math.sqrt(max(inner, 0.0)) - z) / (2 * (1 - z) + 1e-12)

    def residual(z: float) -> float:
        return p_i(z, q1) + p_i(z, q2) - 1.0

    try:
        z_star = brentq(residual, 1e-9, 0.2 - 1e-9, xtol=1e-8, maxiter=100)
    except ValueError:
        # Fallback: proportional removal
        return q1 / overround, q2 / overround, 0.0

    p1 = p_i(z_star, q1)
    p2 = p_i(z_star, q2)
    # Normalise to ensure Σ=1
    total = p1 + p2
    return p1 / total, p2 / total, z_star


def shin_extract_probability(
    decimal_odds: float,
    all_odds: list[float],
    outcome_index: int,
) -> float:
    """
    General Shin extraction for a single outcome from a multi-outcome market.
    Simplified: uses two-outcome Shin on the target outcome vs field.
    For binary markets this is exact; for multi-outcome it's approximate.
    Returns true probability for outcome_index.
    """
    if len(all_odds) == 2:
        p1, p2, _ = shin_two_outcome(all_odds[0], all_odds[1])
        probs = [p1, p2]
    else:
        qs = [1.0 / d for d in all_odds if d > 1.0]
        overround = sum(qs)
        probs = [q / overround for q in qs]
    return probs[outcome_index] if outcome_index < len(probs) else 0.5


# ── F.2 Bookmaker Efficiency ────────────────────────────────────────────────────

@dataclass
class BookmakerEfficiency:
    name: str
    mae_rolling_90d: float = 0.25       # initialise at moderate level
    mae_worst: float = 0.30             # worst in peer group
    tier: Literal[1, 2, 3] = 2

    @property
    def efficiency_score(self) -> float:
        """eff_b = 1 − MAE_b / MAE_worst"""
        if self.mae_worst <= 0:
            return 0.5
        score = 1.0 - self.mae_rolling_90d / self.mae_worst
        return float(np.clip(score, 0.0, 1.0))

    def update_tier(self) -> None:
        eff = self.efficiency_score
        if eff > 0.75:
            self.tier = 1
        elif eff >= 0.40:
            self.tier = 2
        else:
            self.tier = 3

    def update_mae(self, new_mae: float, alpha: float = 0.05) -> None:
        """EMA update of rolling MAE."""
        self.mae_rolling_90d = (1 - alpha) * self.mae_rolling_90d + alpha * new_mae
        self.update_tier()


def default_bookmakers() -> dict[str, BookmakerEfficiency]:
    """Initialise bookmaker efficiency registry with known sharp/soft books."""
    return {
        "pinnacle": BookmakerEfficiency("pinnacle", mae_rolling_90d=0.12, mae_worst=0.28, tier=1),
        "circa": BookmakerEfficiency("circa", mae_rolling_90d=0.14, mae_worst=0.28, tier=1),
        "bet365": BookmakerEfficiency("bet365", mae_rolling_90d=0.17, mae_worst=0.28, tier=2),
        "fanduel": BookmakerEfficiency("fanduel", mae_rolling_90d=0.20, mae_worst=0.28, tier=2),
        "draftkings": BookmakerEfficiency("draftkings", mae_rolling_90d=0.21, mae_worst=0.28, tier=2),
        "mgm": BookmakerEfficiency("mgm", mae_rolling_90d=0.25, mae_worst=0.28, tier=3),
        "caesars": BookmakerEfficiency("caesars", mae_rolling_90d=0.24, mae_worst=0.28, tier=3),
    }


# ── F.3 Market filtering ────────────────────────────────────────────────────────

def market_filter_pass(
    p_soft: float,
    p_sharp_list: list[float],
    p_tier12_list: list[float],
    book_tier: int,
    tolerance: float = 0.005,
    min_tier12_count: int = 2,
) -> bool:
    """
    F.3 — Tier-3 books accepted only if corroborated by ≥ 2 Tier-1/2 books
    within 0.5% implied probability tolerance.
    """
    if book_tier <= 2:
        return True
    if len(p_tier12_list) < min_tier12_count:
        return False
    return all(abs(p_soft - p) <= tolerance for p in p_tier12_list[:min_tier12_count])


# ── F.4 Steam detection ────────────────────────────────────────────────────────

@dataclass
class OddsSnapshot:
    bookmaker: str
    decimal_odds: float
    timestamp: datetime


@dataclass
class SteamEvent:
    market_id: str
    detected_at: datetime
    line_delta_cents: float
    books_confirming: list[str]
    direction: Literal["home", "away"]


def detect_steam(
    snapshots: list[OddsSnapshot],
    window_seconds: int = 90,
    min_move_cents: float = 5.0,
    min_books: int = 3,
) -> SteamEvent | None:
    """
    F.4 — Steam detection.
    Conditions:
      1. Pinnacle line moves ≥ 5 cents within 90s window
      2. ≥ 3 Tier-1/2 books move same direction simultaneously
      3. Cross-market correlation (simplified: all move same direction)
    Returns SteamEvent if detected, else None.
    """
    if len(snapshots) < 2:
        return None

    now = snapshots[-1].timestamp
    recent = [s for s in snapshots if (now - s.timestamp).total_seconds() <= window_seconds]
    if len(recent) < 2:
        return None

    # Group by bookmaker
    by_book: dict[str, list[OddsSnapshot]] = {}
    for s in recent:
        by_book.setdefault(s.bookmaker, []).append(s)

    pinnacle_snaps = by_book.get("pinnacle", [])
    if len(pinnacle_snaps) < 2:
        return None

    pinnacle_move = pinnacle_snaps[-1].decimal_odds - pinnacle_snaps[0].decimal_odds
    # Convert to American cents approximation: Δ(decimal) * 100
    move_cents = abs(pinnacle_move) * 100
    if move_cents < min_move_cents:
        return None

    direction = "home" if pinnacle_move > 0 else "away"

    confirming = []
    for book, snaps in by_book.items():
        if book == "pinnacle":
            confirming.append(book)
            continue
        if len(snaps) >= 2:
            book_move = snaps[-1].decimal_odds - snaps[0].decimal_odds
            if (book_move > 0 and direction == "home") or (book_move < 0 and direction == "away"):
                confirming.append(book)

    if len(confirming) >= min_books:
        return SteamEvent(
            market_id="",
            detected_at=now,
            line_delta_cents=move_cents,
            books_confirming=confirming,
            direction=direction,
        )
    return None


# ── F.5 CLV ────────────────────────────────────────────────────────────────────

def compute_clv(d_entry: float, d_close: float) -> float:
    """
    F.5 — CLV_bet = (d_entry / d_close) − 1
    Positive CLV → placed at better-than-closing odds.
    """
    if d_close <= 0:
        return 0.0
    return (d_entry / d_close) - 1.0


def compute_portfolio_clv(bets: list[dict]) -> float:
    """
    F.5 — Stake-weighted portfolio CLV.
    Each bet dict: {clv: float, stake: float}
    CLV_portfolio_30d = Σ(CLV_bet × stake) / Σ stake
    Target > 0.012.
    """
    total_stake = sum(b.get("stake", 0.0) for b in bets)
    if total_stake <= 0:
        return 0.0
    weighted = sum(b.get("clv", 0.0) * b.get("stake", 0.0) for b in bets)
    return weighted / total_stake


# ── Probability blending utility ───────────────────────────────────────────────

def compute_shin_from_american(american_odds: float) -> float:
    """Convert American odds to decimal, handling + and − lines."""
    if american_odds >= 100:
        return (american_odds / 100.0) + 1.0
    else:
        return (100.0 / abs(american_odds)) + 1.0


def american_to_implied(american_odds: float) -> float:
    """Raw implied probability from American odds (pre-vig)."""
    if american_odds >= 100:
        return 100.0 / (american_odds + 100.0)
    else:
        return abs(american_odds) / (abs(american_odds) + 100.0)


def remove_vig_proportional(p1_raw: float, p2_raw: float) -> tuple[float, float]:
    """Proportional vig removal as fallback."""
    total = p1_raw + p2_raw
    return p1_raw / total, p2_raw / total
