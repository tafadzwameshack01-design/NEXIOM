"""
NEXOM — Dixon–Coles Basketball Adaptation Engine
Section B of the NEXOM v5 specification.

Implements:
  B.1  Attack/defense strength parameters
  B.2  Temporal decay weighting
  B.3  Maximum likelihood estimation via L-BFGS-B
  B.4  Overtime model reference (computed in MC engine)
  B.5  Parameter refresh schedule logic
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
XI: float = 0.0045          # temporal decay constant (days⁻¹)
GAMMA_INIT: float = 1.035   # home court advantage multiplier
PHI_H_INIT: float = 2.0     # NegBin overdispersion home
PHI_A_INIT: float = 2.0     # NegBin overdispersion away
RHO_BOUNDS: tuple = (-0.15, 0.15)
ALPHA_BOUNDS: tuple = (0.5, 2.0)
BETA_BOUNDS: tuple = (0.5, 2.0)
L_AVG_DEFAULT: float = 113.5  # league avg offensive rating pts/100 possessions
MAX_DELTA_T: int = 1022       # cutoff: w(Δt) < 0.01


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class GameRecord:
    """Single game observation for MLE fitting."""
    home_team_id: int
    away_team_id: int
    score_home: int
    score_away: int
    date: datetime
    pace_home: float = 98.0   # possessions per 48 min
    pace_away: float = 98.0


@dataclass
class DCParameters:
    """Full parameter set for Dixon–Coles basketball model."""
    alpha: dict[int, float] = field(default_factory=dict)   # team_id → offensive strength
    beta: dict[int, float] = field(default_factory=dict)    # team_id → defensive resistance
    gamma: float = GAMMA_INIT                                # home court multiplier
    phi_h: float = PHI_H_INIT                               # NegBin overdispersion home
    phi_a: float = PHI_A_INIT                               # NegBin overdispersion away
    rho: float = 0.0                                        # bivariate correction
    last_fitted: datetime = field(default_factory=datetime.utcnow)
    n_games: int = 0
    l_avg: float = L_AVG_DEFAULT


# ── Core functions ─────────────────────────────────────────────────────────────

def temporal_weight(delta_t: int) -> float:
    """
    B.2 — Exponential temporal decay weight.
    w(Δt) = exp(−ξ × Δt), Δt in days.
    Returns 0.0 for Δt > MAX_DELTA_T.
    """
    if delta_t > MAX_DELTA_T:
        return 0.0
    return math.exp(-XI * delta_t)


def expected_efficiency(
    alpha_i: float,
    beta_j: float,
    gamma: float,
    h_ij: int,
    l_avg: float,
) -> float:
    """
    B.1 — λ_{i,j} = α_i × β_j × γ^{H_ij} × L_avg
    Returns expected pts/100 possessions for team i attacking team j.
    """
    return alpha_i * beta_j * (gamma ** h_ij) * l_avg


def close_game_indicator(s_h: int, s_a: int) -> float:
    """
    B.3 — f_close(S_h, S_a): 1 if |S_h - S_a| ≤ 6, else 0.
    Captures close-game regime for bivariate correction.
    """
    return 1.0 if abs(s_h - s_a) <= 6 else 0.0


def bivariate_correction(s_h: int, s_a: int, rho: float) -> float:
    """
    B.3 — τ(S_h, S_a, ρ) = 1 + ρ × f_close(S_h, S_a)
    """
    return 1.0 + rho * close_game_indicator(s_h, s_a)


def negbinom_logpmf(k: int, mu: float, phi: float) -> float:
    """
    Log-PMF of NegativeBinomial(μ, φ) where φ is overdispersion.
    Using parameterisation: n = φ, p = φ/(φ + μ).
    """
    if mu <= 0:
        mu = 1e-6
    p = phi / (phi + mu)
    n = phi
    return nbinom.logpmf(k, n, p)


def weighted_log_likelihood(
    params: np.ndarray,
    games: list[GameRecord],
    team_ids: list[int],
    current_date: datetime,
    l_avg: float,
) -> float:
    """
    B.3 — Weighted negative log-likelihood (negated for minimisation).
    ℓ(Θ, φ_h, φ_a, ρ) = Σ_g w(Δt_g) × [log P(S_h|Λ_h,φ_h) + log P(S_a|Λ_a,φ_a) + log τ(...)]
    """
    n_teams = len(team_ids)
    id_to_idx = {tid: i for i, tid in enumerate(team_ids)}

    # Unpack parameter vector
    # [alpha_0..alpha_N-1, beta_0..beta_N-1, gamma, phi_h, phi_a, rho]
    alphas = np.clip(params[:n_teams], 1e-3, None)
    betas = np.clip(params[n_teams:2 * n_teams], 1e-3, None)
    gamma = max(params[2 * n_teams], 0.5)
    phi_h = max(params[2 * n_teams + 1], 0.1)
    phi_a = max(params[2 * n_teams + 2], 0.1)
    rho = np.clip(params[2 * n_teams + 3], RHO_BOUNDS[0], RHO_BOUNDS[1])

    total_ll = 0.0

    for game in games:
        delta_t = (current_date - game.date).days
        w = temporal_weight(delta_t)
        if w < 1e-6:
            continue

        hi = id_to_idx.get(game.home_team_id)
        ai = id_to_idx.get(game.away_team_id)
        if hi is None or ai is None:
            continue

        pace_g = (game.pace_home + game.pace_away) / 2.0
        lambda_h = expected_efficiency(alphas[hi], betas[ai], gamma, 1, l_avg)
        lambda_a = expected_efficiency(alphas[ai], betas[hi], gamma, 0, l_avg)

        # Convert pts/100 possessions → total expected points
        Lambda_h = lambda_h * (pace_g / 100.0)
        Lambda_a = lambda_a * (pace_g / 100.0)

        ll_h = negbinom_logpmf(game.score_home, Lambda_h, phi_h)
        ll_a = negbinom_logpmf(game.score_away, Lambda_a, phi_a)
        tau = bivariate_correction(game.score_home, game.score_away, rho)
        log_tau = math.log(max(tau, 1e-9))

        total_ll += w * (ll_h + ll_a + log_tau)

    # Identifiability constraint penalty: Σ log(α_i) = 0
    constraint_pen = 1000.0 * (np.sum(np.log(alphas)) ** 2)

    return -(total_ll) + constraint_pen


def fit_dc_parameters(
    games: list[GameRecord],
    current_date: datetime | None = None,
    l_avg: float = L_AVG_DEFAULT,
    max_iter: int = 500,
) -> DCParameters:
    """
    B.3 — Fit Dixon–Coles parameters via L-BFGS-B.
    Returns DCParameters with fitted alpha, beta, gamma, phi_h, phi_a, rho.
    """
    if current_date is None:
        current_date = datetime.utcnow()

    # Collect unique team IDs
    team_ids = sorted(
        set(g.home_team_id for g in games) | set(g.away_team_id for g in games)
    )
    n_teams = len(team_ids)
    if n_teams == 0:
        return DCParameters()

    # Initial parameter vector
    x0 = np.ones(2 * n_teams + 4)
    x0[2 * n_teams] = GAMMA_INIT      # gamma
    x0[2 * n_teams + 1] = PHI_H_INIT  # phi_h
    x0[2 * n_teams + 2] = PHI_A_INIT  # phi_a
    x0[2 * n_teams + 3] = 0.0         # rho

    bounds = (
        [(0.01, 5.0)] * n_teams    # alphas
        + [(0.01, 5.0)] * n_teams  # betas
        + [(0.8, 1.2)]             # gamma
        + [(0.1, 20.0)]            # phi_h
        + [(0.1, 20.0)]            # phi_a
        + [RHO_BOUNDS]             # rho
    )

    try:
        result = minimize(
            weighted_log_likelihood,
            x0,
            args=(games, team_ids, current_date, l_avg),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-5},
        )
        params = result.x
    except Exception as exc:
        logger.warning("L-BFGS-B failed: %s — returning defaults", exc)
        params = x0

    alphas = np.clip(params[:n_teams], ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])
    betas = np.clip(params[n_teams:2 * n_teams], BETA_BOUNDS[0], BETA_BOUNDS[1])
    gamma = float(np.clip(params[2 * n_teams], 0.8, 1.2))
    phi_h = float(max(params[2 * n_teams + 1], 0.1))
    phi_a = float(max(params[2 * n_teams + 2], 0.1))
    rho = float(np.clip(params[2 * n_teams + 3], *RHO_BOUNDS))

    dc = DCParameters(
        alpha={tid: float(alphas[i]) for i, tid in enumerate(team_ids)},
        beta={tid: float(betas[i]) for i, tid in enumerate(team_ids)},
        gamma=gamma,
        phi_h=phi_h,
        phi_a=phi_a,
        rho=rho,
        last_fitted=current_date,
        n_games=len(games),
        l_avg=l_avg,
    )
    return dc


def incremental_newton_update(
    dc: DCParameters,
    new_game: GameRecord,
    current_date: datetime | None = None,
) -> DCParameters:
    """
    B.5 — Single Newton step update after a new game result is ingested.
    Gradient contribution of the new game weighted by w(0) = 1.0.
    Simplified: nudge alpha/beta by gradient * step_size.
    """
    if current_date is None:
        current_date = datetime.utcnow()

    step = 0.005
    h = new_game.home_team_id
    a = new_game.away_team_id
    pace_g = (new_game.pace_home + new_game.pace_away) / 2.0

    alpha_h = dc.alpha.get(h, 1.0)
    alpha_a = dc.alpha.get(a, 1.0)
    beta_h = dc.beta.get(h, 1.0)
    beta_a = dc.beta.get(a, 1.0)

    Lambda_h = expected_efficiency(alpha_h, beta_a, dc.gamma, 1, dc.l_avg) * (pace_g / 100.0)
    Lambda_a = expected_efficiency(alpha_a, beta_h, dc.gamma, 0, dc.l_avg) * (pace_g / 100.0)

    # Gradient of log-likelihood w.r.t. alpha_h: (S_h / Lambda_h - 1) (Poisson approx)
    grad_alpha_h = (new_game.score_home / max(Lambda_h, 1e-6)) - 1.0
    grad_alpha_a = (new_game.score_away / max(Lambda_a, 1e-6)) - 1.0
    grad_beta_a = grad_alpha_h   # beta_a appears in Lambda_h symmetrically
    grad_beta_h = grad_alpha_a

    dc.alpha[h] = float(np.clip(alpha_h + step * grad_alpha_h, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1]))
    dc.alpha[a] = float(np.clip(alpha_a + step * grad_alpha_a, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1]))
    dc.beta[h] = float(np.clip(beta_h + step * grad_beta_h, BETA_BOUNDS[0], BETA_BOUNDS[1]))
    dc.beta[a] = float(np.clip(beta_a + step * grad_beta_a, BETA_BOUNDS[0], BETA_BOUNDS[1]))
    dc.last_fitted = current_date
    dc.n_games += 1
    return dc


def parameter_stability_score(
    dc_current: DCParameters,
    dc_previous: DCParameters,
) -> float:
    """
    K.1 M6 — Σ_i |α_i(t) − α_i(t−7)| / N
    Alert threshold > 0.08.
    """
    team_ids = set(dc_current.alpha.keys()) & set(dc_previous.alpha.keys())
    if not team_ids:
        return 0.0
    total = sum(
        abs(dc_current.alpha[tid] - dc_previous.alpha[tid])
        for tid in team_ids
    )
    return total / len(team_ids)


def generate_synthetic_games(n_teams: int = 10, n_games: int = 200) -> tuple[list[GameRecord], list[int]]:
    """
    Generate synthetic game records for demo/testing.
    Returns (games, team_ids).
    """
    rng = np.random.default_rng(42)
    team_ids = list(range(1, n_teams + 1))
    games: list[GameRecord] = []
    base_date = datetime.utcnow() - timedelta(days=180)

    for _ in range(n_games):
        home_id, away_id = rng.choice(team_ids, 2, replace=False)
        score_h = int(rng.normal(110, 11))
        score_a = int(rng.normal(108, 11))
        score_h = max(70, score_h)
        score_a = max(70, score_a)
        day_offset = int(rng.integers(0, 180))
        games.append(
            GameRecord(
                home_team_id=int(home_id),
                away_team_id=int(away_id),
                score_home=score_h,
                score_away=score_a,
                date=base_date + timedelta(days=day_offset),
                pace_home=float(rng.normal(98, 3)),
                pace_away=float(rng.normal(98, 3)),
            )
        )
    return games, team_ids
