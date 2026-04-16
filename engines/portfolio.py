"""
NEXOM — Portfolio Optimisation Engine (Benter Layer)
Section I of the NEXOM v5 specification.

Implements:
  I.1  Kelly criterion
  I.2  Correlated portfolio Kelly via SLSQP
  I.3  Regime-based scaling
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

Regime = Literal["REGIME_BULL", "REGIME_NORMAL", "REGIME_CAUTIOUS", "REGIME_HALT"]

KELLY_MULTIPLIER_BY_REGIME: dict[Regime, float] = {
    "REGIME_BULL": 0.35,
    "REGIME_NORMAL": 0.30,
    "REGIME_CAUTIOUS": 0.20,
    "REGIME_HALT": 0.0,
}

# Hard exposure caps (I.2)
TOTAL_PORTFOLIO_CAP: float = 0.35    # 35% bankroll
PER_BET_CAP: float = 0.08            # 8% bankroll
PER_GAME_CAP: float = 0.12           # 12% bankroll
PER_BOOK_CAP: float = 0.20           # 20% bankroll
PORTFOLIO_VAR_CAP_FRAC: float = 0.05  # σ_max = 5% × bankroll


@dataclass
class BetCandidate:
    """A candidate bet for portfolio optimisation."""
    bet_id: str
    ev: float           # EV_exec
    p_final: float      # calibrated probability
    decimal_odds: float
    kelly_raw: float    # f* from I.1 (pre-fractional)
    game_id: str
    bookmaker: str
    stake_raw: float    # raw Kelly stake (pre-portfolio optimisation)
    market_type: str = "moneyline"
    outcome_sim: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class PortfolioResult:
    """Output from portfolio optimisation."""
    stakes: dict[str, float]          # bet_id → optimised stake
    fractions: dict[str, float]       # bet_id → fraction of bankroll
    total_exposure: float
    expected_portfolio_ev: float
    portfolio_variance: float
    regime: Regime
    slsqp_converged: bool
    fallback_used: bool


def determine_regime(clv_30d: float, roi_30d: float) -> Regime:
    """I.3 — Regime classification based on 30-day rolling performance."""
    if clv_30d < 0.005 and roi_30d < -0.04:
        return "REGIME_HALT"
    if clv_30d < 0.010 or roi_30d < 0.00:
        return "REGIME_CAUTIOUS"
    if clv_30d > 0.015 and roi_30d > 0.06:
        return "REGIME_BULL"
    return "REGIME_NORMAL"


def compute_correlation_matrix(bets: list[BetCandidate]) -> np.ndarray:
    """
    I.2 — K×K correlation matrix from simulation outcome vectors.
    ρ_{ij} = corr(outcome_i_sim, outcome_j_sim).
    Falls back to identity if sim vectors unavailable.
    """
    K = len(bets)
    C = np.eye(K)
    for i in range(K):
        for j in range(i + 1, K):
            v_i = bets[i].outcome_sim
            v_j = bets[j].outcome_sim
            if len(v_i) > 1 and len(v_j) > 1 and len(v_i) == len(v_j):
                rho = float(np.corrcoef(v_i, v_j)[0, 1])
            elif bets[i].game_id == bets[j].game_id:
                rho = 0.3   # same game: moderate positive correlation
            else:
                rho = 0.05  # different games: low correlation
            C[i, j] = rho
            C[j, i] = rho
    return C


def compute_covariance_matrix(bets: list[BetCandidate], C: np.ndarray) -> np.ndarray:
    """
    I.2 — C_{ij} = ρ_{ij} × √(f*_i × (1−p_i)) × √(f*_j × (1−p_j))
    """
    K = len(bets)
    Cov = np.zeros((K, K))
    for i in range(K):
        fi = bets[i].kelly_raw
        pi = bets[i].p_final
        for j in range(K):
            fj = bets[j].kelly_raw
            pj = bets[j].p_final
            Cov[i, j] = C[i, j] * math.sqrt(max(fi * (1 - pi), 0)) * math.sqrt(max(fj * (1 - pj), 0))
    return Cov


import math


def optimise_portfolio(
    bets: list[BetCandidate],
    bankroll: float,
    regime: Regime = "REGIME_NORMAL",
) -> PortfolioResult:
    """
    I.2 — SLSQP portfolio Kelly optimisation.
    Maximise: Σ f_i × EV_i − 0.5 × Σ_{i,j} f_i × f_j × C_{ij}
    Subject to all exposure constraints.
    Falls back to proportional Kelly if SLSQP fails.
    """
    K = len(bets)
    kelly_mult = KELLY_MULTIPLIER_BY_REGIME[regime]
    sigma_max = (PORTFOLIO_VAR_CAP_FRAC * bankroll) ** 2

    if K == 0:
        return PortfolioResult(
            stakes={}, fractions={}, total_exposure=0.0,
            expected_portfolio_ev=0.0, portfolio_variance=0.0,
            regime=regime, slsqp_converged=True, fallback_used=False,
        )

    # Correlation and covariance
    rho_matrix = compute_correlation_matrix(bets)
    cov_matrix = compute_covariance_matrix(bets, rho_matrix)

    ev_vec = np.array([b.ev for b in bets])
    stake_raw = np.array([b.stake_raw for b in bets])

    # Group bets by game and bookmaker for constraint building
    game_groups: dict[str, list[int]] = {}
    book_groups: dict[str, list[int]] = {}
    for idx, bet in enumerate(bets):
        game_groups.setdefault(bet.game_id, []).append(idx)
        book_groups.setdefault(bet.bookmaker, []).append(idx)

    def neg_objective(f: np.ndarray) -> float:
        growth = float(np.dot(f, ev_vec))
        risk = 0.5 * float(f @ cov_matrix @ f)
        return -(growth - risk)

    def neg_gradient(f: np.ndarray) -> np.ndarray:
        return -(ev_vec - cov_matrix @ f)

    # Bounds: f_i × stake_raw_i ≤ per_bet_cap × bankroll
    per_bet_max = PER_BET_CAP * bankroll
    bounds = []
    for i, bet in enumerate(bets):
        f_max = min(1.0, per_bet_max / max(bet.stake_raw, 1.0))
        bounds.append((0.0, f_max))

    # Constraints
    constraints = []
    # Total portfolio exposure
    constraints.append({
        "type": "ineq",
        "fun": lambda f: TOTAL_PORTFOLIO_CAP * bankroll - float(np.dot(f, stake_raw)),
    })
    # Per-game cap
    for game_id, idxs in game_groups.items():
        def game_constraint(f, _idxs=idxs):
            return PER_GAME_CAP * bankroll - sum(f[i] * stake_raw[i] for i in _idxs)
        constraints.append({"type": "ineq", "fun": game_constraint})
    # Per-bookmaker cap
    for book, idxs in book_groups.items():
        def book_constraint(f, _idxs=idxs):
            return PER_BOOK_CAP * bankroll - sum(f[i] * stake_raw[i] for i in _idxs)
        constraints.append({"type": "ineq", "fun": book_constraint})
    # Portfolio variance cap
    constraints.append({
        "type": "ineq",
        "fun": lambda f: sigma_max - float(f @ cov_matrix @ f),
    })

    x0 = np.clip(np.ones(K) * kelly_mult, 0.0, 1.0)

    converged = False
    fallback = False
    try:
        res = minimize(
            neg_objective,
            x0,
            jac=neg_gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if res.success:
            f_opt = np.clip(res.x, 0.0, 1.0)
            converged = True
        else:
            raise RuntimeError(f"SLSQP did not converge: {res.message}")
    except Exception as exc:
        logger.warning("Portfolio SLSQP failed: %s — using proportional Kelly fallback", exc)
        f_opt = np.clip(np.array([b.kelly_raw * kelly_mult for b in bets]), 0.0, 0.35)
        fallback = True

    stakes = {b.bet_id: float(f_opt[i] * b.stake_raw) for i, b in enumerate(bets)}
    fractions = {b.bet_id: float(f_opt[i] * b.stake_raw / max(bankroll, 1.0)) for i, b in enumerate(bets)}
    total_exp = float(sum(stakes.values()))
    port_ev = float(np.dot(f_opt, ev_vec))
    port_var = float(f_opt @ cov_matrix @ f_opt)

    return PortfolioResult(
        stakes=stakes,
        fractions=fractions,
        total_exposure=total_exp,
        expected_portfolio_ev=port_ev,
        portfolio_variance=port_var,
        regime=regime,
        slsqp_converged=converged,
        fallback_used=fallback,
    )
