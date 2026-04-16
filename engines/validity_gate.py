"""
NEXOM — Final Bet Validity Gate
Section L of the NEXOM v5 specification.

All 10 gates must evaluate to True simultaneously.
Any single False → bet rejected + logged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np


Regime = Literal["REGIME_BULL", "REGIME_NORMAL", "REGIME_CAUTIOUS", "REGIME_HALT"]

MIN_EV_BY_REGIME: dict[Regime, float] = {
    "REGIME_BULL": 0.025,
    "REGIME_NORMAL": 0.030,
    "REGIME_CAUTIOUS": 0.040,
    "REGIME_HALT": 999.0,   # effectively infinite — no betting in HALT
}


@dataclass
class GateInput:
    """All inputs required to evaluate the 10 validity gates."""
    # G1
    ev_exec: float
    # G2
    p_final: float
    p_market: float
    # G3
    ece_30d: float
    n_calibration_samples: int
    calibration_failure_flag: bool
    # G4
    liquidity_factor: float
    # G5 (portfolio)
    portfolio_constraint_passed: bool
    # G6 (flags)
    staleness_flag: bool
    calibration_failure_active: bool
    volatility_spike_active: bool
    category_stop_active: bool
    regime_halt_active: bool
    any_j_stop_active: bool
    # G7 (steam)
    steam_active: bool
    steam_detected_seconds_ago: float
    # G8 (timing)
    timing_factor: float
    # G9 (regime)
    regime: Regime
    # G10 (DC stability)
    dc_m6_shift: float
    dc_m6_threshold: float = 0.08


@dataclass
class GateResult:
    """Result of all 10 gate evaluations."""
    passed: bool
    failed_gates: list[str]
    gate_details: dict[str, dict]


def evaluate_gates(inp: GateInput) -> GateResult:
    """
    Section L — Evaluate all 10 gates sequentially.
    Returns GateResult with passed status and details for each gate.
    """
    results: dict[str, dict] = {}
    failed: list[str] = []

    # ── Gate 1: EV_exec > 0 ───────────────────────────────────────────────────
    g1 = inp.ev_exec > 0.0
    results["GATE_1_EV_POSITIVE"] = {
        "passed": g1,
        "ev_exec": inp.ev_exec,
        "threshold": "> 0.0",
        "description": "Execution-adjusted expected value must be strictly positive",
    }
    if not g1:
        failed.append("GATE_1_EV_POSITIVE")

    # ── Gate 2: Edge minimum ≥ 3 pp ──────────────────────────────────────────
    edge = inp.p_final - inp.p_market
    g2 = edge > 0.030
    results["GATE_2_EDGE_MINIMUM"] = {
        "passed": g2,
        "edge": edge,
        "threshold": "> 0.030",
        "p_final": inp.p_final,
        "p_market": inp.p_market,
        "description": "Minimum 3.0 pp calibrated probability edge over market",
    }
    if not g2:
        failed.append("GATE_2_EDGE_MINIMUM")

    # ── Gate 3: Calibration valid ─────────────────────────────────────────────
    g3 = (inp.ece_30d < 0.060) and (inp.n_calibration_samples >= 500) and (not inp.calibration_failure_flag)
    results["GATE_3_CALIBRATION_VALID"] = {
        "passed": g3,
        "ece_30d": inp.ece_30d,
        "n_samples": inp.n_calibration_samples,
        "failure_flag": inp.calibration_failure_flag,
        "threshold": "ECE < 0.060 AND n ≥ 500 AND no failure flag",
        "description": "Calibration ECE < 0.060, trained on ≥500 observations, no failure flag",
    }
    if not g3:
        failed.append("GATE_3_CALIBRATION_VALID")

    # ── Gate 4: Liquidity sufficient ─────────────────────────────────────────
    g4 = inp.liquidity_factor > 0.50
    results["GATE_4_LIQUIDITY_SUFFICIENT"] = {
        "passed": g4,
        "liquidity_factor": inp.liquidity_factor,
        "threshold": "> 0.50",
        "description": "Intended stake < 50% of estimated available liquidity",
    }
    if not g4:
        failed.append("GATE_4_LIQUIDITY_SUFFICIENT")

    # ── Gate 5: Portfolio constraint ──────────────────────────────────────────
    g5 = inp.portfolio_constraint_passed
    results["GATE_5_PORTFOLIO_CONSTRAINT"] = {
        "passed": g5,
        "description": "Adding this bet does not violate any SLSQP portfolio constraint",
    }
    if not g5:
        failed.append("GATE_5_PORTFOLIO_CONSTRAINT")

    # ── Gate 6: No active flags ───────────────────────────────────────────────
    g6 = not any([
        inp.staleness_flag,
        inp.calibration_failure_active,
        inp.volatility_spike_active,
        inp.category_stop_active,
        inp.regime_halt_active,
        inp.any_j_stop_active,
    ])
    results["GATE_6_NO_ACTIVE_FLAGS"] = {
        "passed": g6,
        "staleness_flag": inp.staleness_flag,
        "calibration_failure": inp.calibration_failure_active,
        "volatility_spike": inp.volatility_spike_active,
        "category_stop": inp.category_stop_active,
        "regime_halt": inp.regime_halt_active,
        "j_stop": inp.any_j_stop_active,
        "description": "No active STALENESS, CALIBRATION_FAILURE, VOLATILITY_SPIKE, CATEGORY_STOP, REGIME_HALT, or J-stop flags",
    }
    if not g6:
        failed.append("GATE_6_NO_ACTIVE_FLAGS")

    # ── Gate 7: Steam clear ───────────────────────────────────────────────────
    if not inp.steam_active:
        g7 = True
        steam_note = "no_steam"
    else:
        if inp.steam_detected_seconds_ago < 300:
            g7 = inp.ev_exec > 0.055   # elevated threshold during steam
            steam_note = f"steam_active_{inp.steam_detected_seconds_ago:.0f}s_ago"
        else:
            g7 = True
            steam_note = "steam_expired"
    results["GATE_7_STEAM_CLEAR"] = {
        "passed": g7,
        "steam_active": inp.steam_active,
        "steam_seconds_ago": inp.steam_detected_seconds_ago,
        "steam_note": steam_note,
        "elevated_threshold_applied": inp.steam_active and inp.steam_detected_seconds_ago < 300,
        "description": "No steam active OR steam detected but EV > 0.055",
    }
    if not g7:
        failed.append("GATE_7_STEAM_CLEAR")

    # ── Gate 8: Timing valid ──────────────────────────────────────────────────
    g8 = inp.timing_factor > 0.35
    results["GATE_8_TIMING_VALID"] = {
        "passed": g8,
        "timing_factor": inp.timing_factor,
        "threshold": "> 0.35",
        "description": "Bet entered within viable execution window (timing_factor > 0.35)",
    }
    if not g8:
        failed.append("GATE_8_TIMING_VALID")

    # ── Gate 9: Execution value meets regime threshold ────────────────────────
    min_ev = MIN_EV_BY_REGIME.get(inp.regime, 0.030)
    g9 = inp.ev_exec > min_ev
    results["GATE_9_EXECUTION_VALUE"] = {
        "passed": g9,
        "ev_exec": inp.ev_exec,
        "regime": inp.regime,
        "min_ev_threshold": min_ev,
        "description": f"EV_exec > {min_ev:.3f} in {inp.regime}",
    }
    if not g9:
        failed.append("GATE_9_EXECUTION_VALUE")

    # ── Gate 10: DC parameter stable ─────────────────────────────────────────
    g10 = inp.dc_m6_shift < inp.dc_m6_threshold
    results["GATE_10_DC_PARAMETER_STABLE"] = {
        "passed": g10,
        "m6_shift": inp.dc_m6_shift,
        "threshold": inp.dc_m6_threshold,
        "description": "Dixon–Coles M6 parameter shift below drift alert threshold",
    }
    if not g10:
        failed.append("GATE_10_DC_PARAMETER_STABLE")

    all_passed = len(failed) == 0
    return GateResult(passed=all_passed, failed_gates=failed, gate_details=results)
