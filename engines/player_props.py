"""
NEXOM — Player Prop Model
Section D of the NEXOM v5 specification.

Implements:
  D.1  Per-player per-stat projection
  D.2  Gaussian copula correlation structure
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

Stat = Literal["PTS", "REB", "AST", "STL", "BLK", "3PM"]

# D.2 — NegBin overdispersion by stat (empirically calibrated)
PHI_BY_STAT: dict[str, float] = {
    "PTS": 2.1,
    "REB": 1.8,
    "AST": 1.9,
    "STL": 1.5,
    "BLK": 1.4,
    "3PM": 1.6,
}

# Position-average rates (pts/min baseline)
LEAGUE_AVG_RATES: dict[str, float] = {
    "PTS": 0.55,
    "REB": 0.22,
    "AST": 0.18,
    "STL": 0.035,
    "BLK": 0.025,
    "3PM": 0.10,
}


@dataclass
class PlayerData:
    player_id: int
    name: str
    position: str
    season_avg_minutes: float = 30.0
    is_b2b: bool = False
    is_load_management_risk: bool = False
    days_rest: int = 2
    per_min_rates: dict[str, float] = field(default_factory=dict)    # stat → rate per minute
    game_log: list[dict] = field(default_factory=list)               # list of {stat: value, ...}
    usage_pct: float = 0.20
    baseline_usage_pct: float = 0.20


@dataclass
class LineupContext:
    home_player_ids: list[int]
    away_player_ids: list[int]
    opponent_def_rates: dict[str, float] = field(default_factory=dict)   # stat → opp allowed rate vs position


@dataclass
class PropProjection:
    player_id: int
    name: str
    stat: str
    projection: float
    minutes_mean: float
    minutes_std: float = 4.5
    matchup_factor: float = 1.0
    usage_factor: float = 1.0
    rest_factor: float = 1.0
    lineup_factor: float = 1.0


def minutes_projection(player: PlayerData) -> tuple[float, float]:
    """D.1 — minutes_projection ~ TruncatedNormal(μ_min, σ_min²,[0,48])"""
    mu = player.season_avg_minutes
    if player.is_b2b:
        mu -= 2.0
    if player.is_load_management_risk:
        mu -= 1.0
    mu = float(np.clip(mu, 0.0, 48.0))
    return mu, 4.5


def rest_factor(days_rest: int) -> float:
    """D.1 — rest_factor based on days between games."""
    if days_rest >= 4:
        return 1.02
    if days_rest == 1:
        return 0.97
    return 1.00


def matchup_factor(
    stat: str,
    opp_def_rate: float,
    league_avg_def_rate: float | None = None,
) -> float:
    """D.1 — matchup_factor = opp_def_rate / L_avg_def_rate"""
    avg = league_avg_def_rate or LEAGUE_AVG_RATES.get(stat, 1.0)
    if avg <= 0:
        return 1.0
    return float(opp_def_rate / avg)


def usage_factor(player: PlayerData) -> float:
    """D.1 — usage_factor = USG%(i) / baseline_USG%(i)"""
    if player.baseline_usage_pct <= 0:
        return 1.0
    return float(player.usage_pct / player.baseline_usage_pct)


def bayesian_rate(
    player: PlayerData,
    stat: str,
    n_games_recent: int = 15,
) -> float:
    """
    D.1 — Bayesian posterior mean for per-minute rate.
    Prior: positional league avg (σ_prior = 0.3 × L_avg_s)
    Likelihood: rolling 15-game per-minute rate.
    Normal-normal conjugate update.
    """
    L_avg = LEAGUE_AVG_RATES.get(stat, 0.5)
    prior_mean = L_avg
    prior_var = (0.3 * L_avg) ** 2

    if not player.game_log:
        return prior_mean

    recent = player.game_log[-n_games_recent:]
    rates = []
    for game in recent:
        mins = game.get("minutes", player.season_avg_minutes)
        val = game.get(stat, 0.0)
        if mins > 0:
            rates.append(val / mins)

    if not rates:
        return prior_mean

    n = len(rates)
    obs_mean = float(np.mean(rates))
    obs_var = float(np.var(rates)) if n > 1 else prior_var
    likelihood_var = obs_var / max(n, 1)

    # Normal-normal conjugate update
    posterior_var = 1.0 / (1.0 / prior_var + 1.0 / max(likelihood_var, 1e-9))
    posterior_mean = posterior_var * (prior_mean / prior_var + obs_mean / max(likelihood_var, 1e-9))
    return float(max(posterior_mean, 0.0))


def project_stat(
    player: PlayerData,
    stat: str,
    lineup_context: LineupContext | None = None,
    opp_def_rate: float | None = None,
) -> PropProjection:
    """D.1 — Full per-player per-stat projection."""
    rate = bayesian_rate(player, stat)
    mu_min, sigma_min = minutes_projection(player)
    rf = rest_factor(player.days_rest)
    uf = usage_factor(player)

    if opp_def_rate is not None:
        mf = matchup_factor(stat, opp_def_rate)
    else:
        mf = 1.0

    lf = 1.0   # lineup_factor: simplified to 1.0 without NMF synergy matrix

    projection = rate * mu_min * mf * uf * rf * lf

    return PropProjection(
        player_id=player.player_id,
        name=player.name,
        stat=stat,
        projection=float(max(projection, 0.0)),
        minutes_mean=mu_min,
        matchup_factor=mf,
        usage_factor=uf,
        rest_factor=rf,
        lineup_factor=lf,
    )


def gaussian_copula_simulate(
    projections: dict[str, float],
    sigma_matrix: np.ndarray,
    n_samples: int = 10000,
) -> np.ndarray:
    """
    D.2 — Gaussian copula simulation for joint stat distribution.
    sigma_matrix: 6×6 correlation matrix (Ledoit-Wolf shrunk).
    Returns array of shape (n_samples, 6) with simulated stat values.
    Order: PTS, REB, AST, STL, BLK, 3PM
    """
    stats_order = ["PTS", "REB", "AST", "STL", "BLK", "3PM"]
    n_stats = len(stats_order)

    # Z ~ N(0, Σ)
    try:
        Z = np.random.multivariate_normal(np.zeros(n_stats), sigma_matrix, size=n_samples)
    except np.linalg.LinAlgError:
        Z = np.random.normal(0, 1, (n_samples, n_stats))

    # U = Φ(Z) — uniform marginals
    U = stats.norm.cdf(Z)

    # X_i = F_i^{-1}(U_i) via NegBin inverse CDF
    X = np.zeros((n_samples, n_stats))
    for j, stat in enumerate(stats_order):
        mu = max(projections.get(stat, 0.0), 0.01)
        phi = PHI_BY_STAT[stat]
        p_nb = phi / (phi + mu)
        X[:, j] = stats.nbinom.ppf(U[:, j], phi, p_nb).astype(float)

    return X


def estimate_correlation_matrix(game_log: list[dict]) -> np.ndarray:
    """
    D.2 — Estimate 6×6 correlation matrix via Ledoit-Wolf shrinkage.
    Requires ≥ 30 game observations; falls back to positional average.
    """
    stats_order = ["PTS", "REB", "AST", "STL", "BLK", "3PM"]
    if len(game_log) < 30:
        # Positional average (moderate positive correlations)
        Sigma = np.eye(6)
        Sigma[0, 2] = Sigma[2, 0] = 0.3   # PTS–AST
        Sigma[0, 1] = Sigma[1, 0] = 0.15  # PTS–REB
        return Sigma

    data = np.array([[g.get(s, 0.0) for s in stats_order] for g in game_log])
    lw = LedoitWolf(assume_centered=False)
    lw.fit(data)
    cov = lw.covariance_
    # Convert to correlation matrix
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def prop_over_probability(
    projection: float,
    line: float,
    phi: float = 2.0,
    n_samples: int = 10000,
) -> float:
    """Monte Carlo P(stat > line) using NegBin distribution."""
    mu = max(projection, 0.01)
    p_nb = phi / (phi + mu)
    samples = np.random.negative_binomial(phi, p_nb, size=n_samples).astype(float)
    return float(np.clip(np.mean(samples > line), 0.001, 0.999))


def generate_demo_player(player_id: int = 1, name: str = "Demo Player") -> PlayerData:
    """Generate a demo player with synthetic game log."""
    rng = np.random.default_rng(player_id)
    game_log = []
    for _ in range(40):
        pts = float(rng.negative_binomial(5, 0.20))
        reb = float(rng.negative_binomial(3, 0.40))
        ast = float(rng.negative_binomial(3, 0.45))
        stl = float(rng.negative_binomial(2, 0.50))
        blk = float(rng.negative_binomial(2, 0.57))
        tpm = float(rng.negative_binomial(2, 0.45))
        mins = float(rng.normal(30, 4))
        game_log.append({
            "PTS": pts, "REB": reb, "AST": ast, "STL": stl,
            "BLK": blk, "3PM": tpm, "minutes": max(mins, 5.0),
        })
    return PlayerData(
        player_id=player_id,
        name=name,
        position="SG",
        season_avg_minutes=30.0,
        is_b2b=False,
        is_load_management_risk=False,
        days_rest=2,
        per_min_rates={},
        game_log=game_log,
        usage_pct=0.22,
        baseline_usage_pct=0.22,
    )
