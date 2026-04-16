"""
NEXOM — Vectorised Monte Carlo Engine
Section E of the NEXOM v5 specification.

Vectorisation requirements (E.2):
  - No Python loop over N simulations
  - All operations across N-axis use NumPy array ops
  - Gumbel-max trick for multinomial sampling
  - N_pre=100,000 | N_live=25,000
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
N_PRE: int = 100_000
N_LIVE: int = 25_000
MAX_POSSESSIONS: int = 250
SIGMA_PACE: float = 3.5          # std dev of pace deviation (possessions/48min)
PACE_MIN: float = 80.0
PACE_MAX: float = 115.0

# Outcome indices — C.3
#  0=TO  1=2PM  2=2PA_FT  3=3PM  4=3PA_FT  5=FT_ONLY  6=OREB  7=END
OUTCOME_POINTS = np.array([0, 2, 2, 3, 3, 1, 0, 0], dtype=np.float64)
# Approximate FT points for 2PA_FT and 3PA_FT
FT_POINTS = np.array([0, 0, 1.4, 0, 2.1, 1.4, 0, 0], dtype=np.float64)
TOTAL_POINTS_PER_OUTCOME = OUTCOME_POINTS + FT_POINTS

# Default transition probabilities (league-average baseline)
# Calibrated from NBA 2016–2024 possession outcomes
DEFAULT_OUTCOME_PROBS = np.array(
    [0.13, 0.35, 0.08, 0.20, 0.04, 0.05, 0.12, 0.03], dtype=np.float64
)
DEFAULT_OUTCOME_PROBS /= DEFAULT_OUTCOME_PROBS.sum()

# Seconds per possession (approx 14s)
SECONDS_PER_POSSESSION: float = 14.0


@dataclass
class MCInput:
    """Inputs for the Monte Carlo engine."""
    alpha_home_mean: float = 1.0
    alpha_away_mean: float = 1.0
    beta_home_mean: float = 1.0
    beta_away_mean: float = 1.0
    alpha_home_std: float = 0.05
    alpha_away_std: float = 0.05
    beta_home_std: float = 0.05
    beta_away_std: float = 0.05
    gamma: float = 1.035
    pace: float = 98.0            # expected possessions per 48 min
    l_avg: float = 113.5          # league avg pts/100 possessions
    spread: float = 0.0           # spread line for P(cover) computation
    total_line: float = 220.0     # O/U line
    fatigue_home: float = 0.0
    fatigue_away: float = 0.0
    # Current game state for live mode
    score_home: int = 0
    score_away: int = 0
    time_remaining_total: float = 2880.0  # seconds (4 × 720)
    n_simulations: int = N_PRE
    rho: float = 0.0


@dataclass
class MCOutput:
    """Outputs from the Monte Carlo engine."""
    p_home_win: float
    p_cover_spread: float
    p_over_total: float
    p_overtime: float
    p_home_cover_alt: float
    mean_total: float
    std_total: float
    mean_margin: float
    std_margin: float
    n_simulations: int
    elapsed_seconds: float


def _gumbel_max_sample(log_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    E.3 Step 3 — Gumbel-max trick for vectorised multinomial sampling.
    log_probs shape: (N, K)
    Returns argmax shape: (N,)
    """
    gumbel = rng.gumbel(0, 1, log_probs.shape)
    return np.argmax(gumbel + log_probs, axis=1)


def _compute_outcome_probs(
    alpha_off: np.ndarray,
    beta_def: np.ndarray,
    gamma: float,
    h_indicator: float,
    l_avg: float,
    fatigue: np.ndarray,
    pace_sim: np.ndarray,
) -> np.ndarray:
    """
    Vectorised outcome probability computation.
    Modulates league-average probs by team strengths and fatigue.

    alpha_off, beta_def, pace_sim: shape (N,)
    Returns shape (N, 8) normalised outcome probs.
    """
    N = len(alpha_off)
    # Relative efficiency vs league avg
    efficiency = (alpha_off * beta_def * (gamma ** h_indicator)) / 1.0
    # Fatigue degrades offensive efficiency
    eff_adj = efficiency * (1.0 - 0.05 * fatigue)
    # Modulate default probs: higher efficiency → more made shots, fewer TOs
    probs = np.tile(DEFAULT_OUTCOME_PROBS, (N, 1))  # (N, 8)
    # Scale 2PM and 3PM by efficiency ratio, compensate TO
    scale = np.clip(eff_adj, 0.6, 1.5)[:, None]
    probs[:, 1] *= scale[:, 0]  # 2PM
    probs[:, 3] *= scale[:, 0]  # 3PM
    probs[:, 0] /= scale[:, 0]  # TO decreases when efficient
    # Renormalise
    row_sums = probs.sum(axis=1, keepdims=True)
    probs /= row_sums
    return probs


def run_simulation(inp: MCInput) -> MCOutput:
    """
    E.3 — Full vectorised Monte Carlo simulation.
    Zero Python loops over the N-simulation axis.
    Returns MCOutput with all probability estimates.
    """
    t_start = time.perf_counter()
    N = inp.n_simulations
    rng = np.random.default_rng()

    # Step 1: Sample attack/defense parameters (N,)
    alpha_h = np.clip(rng.normal(inp.alpha_home_mean, inp.alpha_home_std, N), 1e-3, None)
    alpha_a = np.clip(rng.normal(inp.alpha_away_mean, inp.alpha_away_std, N), 1e-3, None)
    beta_h = np.clip(rng.normal(inp.beta_home_mean, inp.beta_home_std, N), 1e-3, None)
    beta_a = np.clip(rng.normal(inp.beta_away_mean, inp.beta_away_std, N), 1e-3, None)

    # Step 2: Sample pace (N,)
    pace_sim = np.clip(rng.normal(inp.pace, SIGMA_PACE, N), PACE_MIN, PACE_MAX)

    # Initialise game state arrays (N,)
    score_home = np.full(N, float(inp.score_home))
    score_away = np.full(N, float(inp.score_away))
    time_remaining = np.full(N, inp.time_remaining_total)
    home_possesses = rng.integers(0, 2, N).astype(bool)  # random first possession

    # Fatigue arrays
    fatigue_h = np.full(N, inp.fatigue_home)
    fatigue_a = np.full(N, inp.fatigue_away)

    # OT tracking: tied at regulation end
    regulation_end_tied = np.zeros(N, dtype=bool)

    # Seconds per possession scales with pace
    secs_per_poss = (48 * 60) / pace_sim  # seconds per possession in regulation

    for _step in range(MAX_POSSESSIONS):
        # Mask: which sims still have time
        active = time_remaining > 0
        if not active.any():
            break

        # Offense/defense alpha-beta for this possession
        alpha_off = np.where(home_possesses, alpha_h, alpha_a)
        beta_def = np.where(home_possesses, beta_a, beta_h)
        fatigue_off = np.where(home_possesses, fatigue_h, fatigue_a)
        h_ind = home_possesses.astype(float)

        # Outcome probabilities (N, 8)
        probs = _compute_outcome_probs(alpha_off, beta_def, inp.gamma, 1.0, inp.l_avg, fatigue_off, pace_sim)
        log_probs = np.log(probs + 1e-9)

        # Sample outcomes via Gumbel-max
        outcomes = _gumbel_max_sample(log_probs, rng)  # (N,)

        # Points scored this possession
        pts = TOTAL_POINTS_PER_OUTCOME[outcomes]  # (N,)
        pts = pts * active.astype(float)           # zero out inactive sims

        # Accumulate scores
        score_home += pts * home_possesses.astype(float)
        score_away += pts * (~home_possesses).astype(float)

        # Advance time
        time_remaining -= secs_per_poss
        time_remaining = np.maximum(time_remaining, 0.0)

        # Mark regulation end tied (before OT extension)
        # Check sims that just crossed 0
        just_ended = active & (time_remaining <= 0)
        regulation_end_tied |= just_ended & (score_home == score_away)

        # Flip possession (simple model; OREB handled by effective pts model)
        flip = outcomes != 6  # OREB keeps possession
        home_possesses = np.where(flip, ~home_possesses, home_possesses)

        # Fatigue increment per possession
        fatigue_h = np.clip(fatigue_h + 0.001, 0.0, 1.0)
        fatigue_a = np.clip(fatigue_a + 0.001, 0.0, 1.0)

    # OT extension: run 1 OT period (5 min = 300s) for tied sims
    ot_sims = regulation_end_tied.copy()
    if ot_sims.any():
        ot_time = np.where(ot_sims, 300.0, 0.0)
        for _ot in range(50):  # max 50 OT possession steps
            active_ot = ot_time > 0
            if not active_ot.any():
                break
            alpha_off = np.where(home_possesses, alpha_h, alpha_a)
            beta_def = np.where(home_possesses, beta_a, beta_h)
            fatigue_off = np.where(home_possesses, fatigue_h, fatigue_a)
            probs = _compute_outcome_probs(alpha_off, beta_def, inp.gamma, 1.0, inp.l_avg, fatigue_off, pace_sim)
            log_probs = np.log(probs + 1e-9)
            outcomes = _gumbel_max_sample(log_probs, rng)
            pts = TOTAL_POINTS_PER_OUTCOME[outcomes] * active_ot.astype(float)
            score_home += pts * home_possesses.astype(float)
            score_away += pts * (~home_possesses).astype(float)
            ot_time -= secs_per_poss
            ot_time = np.maximum(ot_time, 0.0)
            flip = outcomes != 6
            home_possesses = np.where(flip, ~home_possesses, home_possesses)

    # Step 5: Market outcome probabilities
    winner = (score_home > score_away).astype(int)
    margin = score_home - score_away
    total = score_home + score_away

    p_home_win = float(np.clip(np.mean(winner), 0.001, 0.999))
    p_cover = float(np.clip(np.mean(margin > inp.spread), 0.001, 0.999))
    p_over = float(np.clip(np.mean(total > inp.total_line), 0.001, 0.999))
    p_ot = float(np.clip(np.mean(regulation_end_tied), 0.001, 0.999))
    p_home_alt = float(np.clip(np.mean(margin > (inp.spread + 3)), 0.001, 0.999))

    elapsed = time.perf_counter() - t_start

    return MCOutput(
        p_home_win=p_home_win,
        p_cover_spread=p_cover,
        p_over_total=p_over,
        p_overtime=p_ot,
        p_home_cover_alt=p_home_alt,
        mean_total=float(np.mean(total)),
        std_total=float(np.std(total)),
        mean_margin=float(np.mean(margin)),
        std_margin=float(np.std(margin)),
        n_simulations=N,
        elapsed_seconds=elapsed,
    )


def get_score_distribution(inp: MCInput, bins: int = 50) -> dict[str, Any]:
    """
    Run simulation and return score distributions for visualisation.
    Uses reduced N for speed in UI context.
    """
    inp_light = MCInput(**{**inp.__dict__, "n_simulations": min(inp.n_simulations, 10000)})
    N = inp_light.n_simulations
    rng = np.random.default_rng(0)

    alpha_h = np.clip(rng.normal(inp_light.alpha_home_mean, inp_light.alpha_home_std, N), 1e-3, None)
    alpha_a = np.clip(rng.normal(inp_light.alpha_away_mean, inp_light.alpha_away_std, N), 1e-3, None)
    beta_h = np.clip(rng.normal(inp_light.beta_home_mean, inp_light.beta_home_std, N), 1e-3, None)
    beta_a = np.clip(rng.normal(inp_light.beta_away_mean, inp_light.beta_away_std, N), 1e-3, None)
    pace_sim = np.clip(rng.normal(inp_light.pace, SIGMA_PACE, N), PACE_MIN, PACE_MAX)

    score_home = np.zeros(N)
    score_away = np.zeros(N)
    time_remaining = np.full(N, inp_light.time_remaining_total)
    home_possesses = rng.integers(0, 2, N).astype(bool)
    fatigue_h = np.full(N, inp_light.fatigue_home)
    fatigue_a = np.full(N, inp_light.fatigue_away)
    secs_per_poss = (48 * 60) / pace_sim

    for _step in range(MAX_POSSESSIONS):
        active = time_remaining > 0
        if not active.any():
            break
        alpha_off = np.where(home_possesses, alpha_h, alpha_a)
        beta_def = np.where(home_possesses, beta_a, beta_h)
        fatigue_off = np.where(home_possesses, fatigue_h, fatigue_a)
        probs = _compute_outcome_probs(alpha_off, beta_def, inp_light.gamma, 1.0, inp_light.l_avg, fatigue_off, pace_sim)
        log_probs = np.log(probs + 1e-9)
        outcomes = _gumbel_max_sample(log_probs, rng)
        pts = TOTAL_POINTS_PER_OUTCOME[outcomes] * active.astype(float)
        score_home += pts * home_possesses.astype(float)
        score_away += pts * (~home_possesses).astype(float)
        time_remaining -= secs_per_poss
        time_remaining = np.maximum(time_remaining, 0.0)
        flip = outcomes != 6
        home_possesses = np.where(flip, ~home_possesses, home_possesses)

    margin = score_home - score_away
    total = score_home + score_away
    margin_hist, margin_bins = np.histogram(margin, bins=bins)
    total_hist, total_bins = np.histogram(total, bins=bins)

    return {
        "margin_hist": margin_hist.tolist(),
        "margin_bins": [(margin_bins[i] + margin_bins[i + 1]) / 2 for i in range(len(margin_hist))],
        "total_hist": total_hist.tolist(),
        "total_bins": [(total_bins[i] + total_bins[i + 1]) / 2 for i in range(len(total_hist))],
        "score_home": score_home.tolist()[:500],   # sample for scatter
        "score_away": score_away.tolist()[:500],
    }
