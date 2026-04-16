"""
NEXOM — Execution Microstructure Engine (Bloom Layer)
Section G of the NEXOM v5 specification.

Implements:
  G.1  Execution value: EV_exec
  G.2  Latency model (LogNormal)
  G.3  Sharp/soft divergence signal
  G.4  Execution timing strategy
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np


# ── G.1 Execution value ────────────────────────────────────────────────────────

@dataclass
class ExecutionParams:
    """Parameters for EV_exec computation."""
    p_final: float          # calibrated model probability
    p_market: float         # Shin-extracted market probability
    intended_stake: float   # dollar stake
    estimated_liquidity: float  # available liquidity at current odds
    t_delay_minutes: float  # elapsed minutes since signal generation
    book_stake_limit: float = 5000.0  # historical max accepted stake
    bookmaker: str = "pinnacle"
    decimal_odds: float = 1.90


def compute_edge_net(p_final: float, p_market: float, min_edge: float = 0.030) -> float:
    """G.1 — edge_net = p_final − p_market. Must be > 0.030."""
    return p_final - p_market


def compute_liquidity_factor(intended_stake: float, estimated_liquidity: float) -> float:
    """G.1 — liquidity_factor = min(1.0, est_liquidity / (3 × intended_stake))"""
    if intended_stake <= 0:
        return 0.0
    return float(min(1.0, estimated_liquidity / (3.0 * intended_stake + 1e-9)))


def compute_timing_factor(t_delay_minutes: float) -> float:
    """G.1 — timing_factor = exp(−0.08 × t_delay_minutes). Valid when > 0.35."""
    return float(math.exp(-0.08 * t_delay_minutes))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_fill_probability(intended_stake: float, book_stake_limit: float) -> float:
    """G.1 — fill_prob = sigmoid(−2.5 × (stake/limit − 0.8))"""
    ratio = intended_stake / max(book_stake_limit, 1.0)
    return float(sigmoid(-2.5 * (ratio - 0.8)))


def compute_latency_adjustment(
    odds_offered: float,
    mu_b: float,
    sigma_b: float,
    drift_rate_b: float,
) -> tuple[float, float]:
    """
    G.2 — Latency adjustment.
    Draws latency from LogNormal(μ_b, σ_b²), computes odds degradation.
    Returns (latency_adjustment, latency_ms_sample).
    """
    latency_ms = float(np.random.lognormal(mu_b, sigma_b))
    odds_received = odds_offered - drift_rate_b * latency_ms / 1000.0
    if odds_offered <= 1.0:
        return 1.0, latency_ms
    adj = max(0.0, (odds_received - 1.0) / (odds_offered - 1.0))
    return float(adj), latency_ms


def compute_ev_exec(params: ExecutionParams) -> dict:
    """
    G.1 — EV_exec = edge_net × liquidity_factor × timing_factor × fill_probability × latency_adjustment
    Returns all components + final EV_exec.
    """
    edge_net = compute_edge_net(params.p_final, params.p_market)
    liquidity_factor = compute_liquidity_factor(params.intended_stake, params.estimated_liquidity)
    timing_factor = compute_timing_factor(params.t_delay_minutes)
    fill_prob = compute_fill_probability(params.intended_stake, params.book_stake_limit)

    # Latency params by bookmaker
    latency_params = {
        "pinnacle": (5.5, 0.4),
        "circa": (5.5, 0.4),
        "bet365": (5.8, 0.5),
        "default": (5.8, 0.6),
    }
    mu_b, sigma_b = latency_params.get(params.bookmaker, latency_params["default"])
    drift_rate_b = 0.002  # decimal odds per second
    lat_adj, lat_ms = compute_latency_adjustment(params.decimal_odds, mu_b, sigma_b, drift_rate_b)

    ev_exec = edge_net * liquidity_factor * timing_factor * fill_prob * lat_adj

    return {
        "ev_exec": ev_exec,
        "edge_net": edge_net,
        "liquidity_factor": liquidity_factor,
        "timing_factor": timing_factor,
        "fill_probability": fill_prob,
        "latency_adjustment": lat_adj,
        "latency_ms": lat_ms,
    }


# ── G.3 Sharp/Soft divergence ──────────────────────────────────────────────────

def sharp_soft_divergence(
    p_sharp: float,
    p_soft_list: list[float],
    min_soft_books: int = 3,
) -> dict:
    """
    G.3 — Δ_{sharp-soft} = p_sharp − avg(p_soft)
    Returns signal and position size modifier.
    """
    if len(p_soft_list) < min_soft_books:
        return {"delta": 0.0, "signal": "neutral", "size_modifier": 1.0, "valid": False}

    p_soft_avg = float(np.mean(p_soft_list))
    delta = p_sharp - p_soft_avg

    if delta > 0.04:
        signal = "sharp_corroborating"
        size_modifier = 1.2
    elif delta < -0.04:
        signal = "sharp_counter"
        size_modifier = 0.7
    else:
        signal = "neutral"
        size_modifier = 1.0

    return {
        "delta": delta,
        "p_sharp": p_sharp,
        "p_soft_avg": p_soft_avg,
        "signal": signal,
        "size_modifier": size_modifier,
        "valid": True,
    }


# ── G.4 Execution timing ───────────────────────────────────────────────────────

def execution_window_status(
    tip_off_time: datetime,
    current_time: datetime | None = None,
    is_live: bool = False,
    signal_generated_at: datetime | None = None,
) -> dict:
    """
    G.4 — Determine if we're within a valid execution window.
    Pre-game windows: T−90→T−30 (primary), T−10→T−5 (secondary).
    Never execute within T−2.
    Live: must execute within 12 seconds of signal.
    """
    if current_time is None:
        current_time = datetime.utcnow()

    if is_live:
        if signal_generated_at is None:
            return {"valid": False, "reason": "no_signal_time"}
        elapsed_s = (current_time - signal_generated_at).total_seconds()
        timing_factor = compute_timing_factor(elapsed_s / 60.0)
        return {
            "valid": elapsed_s <= 12.0 and timing_factor > 0.38,
            "elapsed_seconds": elapsed_s,
            "timing_factor": timing_factor,
            "window": "live",
            "reason": "ok" if elapsed_s <= 12.0 else "too_late",
        }

    minutes_to_tip = (tip_off_time - current_time).total_seconds() / 60.0

    if minutes_to_tip < 2:
        return {"valid": False, "window": "blackout", "minutes_to_tip": minutes_to_tip,
                "reason": "within_2min_of_tipoff"}
    if 5 <= minutes_to_tip <= 10:
        return {"valid": True, "window": "secondary_pregame", "minutes_to_tip": minutes_to_tip,
                "reason": "secondary_window"}
    if 30 <= minutes_to_tip <= 90:
        return {"valid": True, "window": "primary_pregame", "minutes_to_tip": minutes_to_tip,
                "reason": "primary_window"}

    return {"valid": False, "window": "outside_windows", "minutes_to_tip": minutes_to_tip,
            "reason": "outside_execution_windows"}


# ── Kelly criterion ────────────────────────────────────────────────────────────

def kelly_fraction(p_final: float, decimal_odds: float, cap: float = 0.25) -> float:
    """
    I.1 — f* = (p × d − 1) / (d − 1)
    Capped at 0.25.
    """
    if decimal_odds <= 1.0:
        return 0.0
    f = (p_final * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return float(np.clip(f, 0.0, cap))


def fractional_kelly_stake(
    p_final: float,
    decimal_odds: float,
    bankroll: float,
    kelly_multiplier: float = 0.30,
) -> float:
    """I.1 — actual stake = f* × kelly_fraction_multiplier × bankroll"""
    f = kelly_fraction(p_final, decimal_odds)
    return float(f * kelly_multiplier * bankroll)
