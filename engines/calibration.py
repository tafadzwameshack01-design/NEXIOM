"""
NEXOM — Calibration Engine (Silver Layer)
Section H of the NEXOM v5 specification.

Implements:
  H.1  Probability blending (α·p_model + β·p_market + γ·p_prior)
  H.2  Isotonic regression calibration
  H.3  Platt scaling per market type
  H.4  Final calibrated probability selection
  H.5  ECE miscalibration rejection
  H.6  Combined Brier + log-loss objective
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

MarketType = Literal["moneyline", "spread", "total", "player_prop"]
MIN_CALIBRATION_SAMPLES: int = 500


@dataclass
class CalibrationWeights:
    """H.1 blending weights α + β + γ = 1."""
    alpha: float = 0.45   # model weight
    beta: float = 0.40    # market weight
    gamma: float = 0.15   # prior weight

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.beta, self.gamma])


@dataclass
class PlattScaler:
    """H.3 per-market Platt scaling: p = sigmoid(A·p_blend + B)."""
    market_type: MarketType
    A: float = 1.0
    B: float = 0.0
    trained: bool = False

    def predict(self, p_blend: float) -> float:
        x = self.A * p_blend + self.B
        return float(1.0 / (1.0 + math.exp(-x)))


@dataclass
class CalibrationState:
    """Full calibration state per market type."""
    weights: CalibrationWeights = field(default_factory=CalibrationWeights)
    isotonic: IsotonicRegression | None = None
    isotonic_trained: bool = False
    platt: dict[MarketType, PlattScaler] = field(default_factory=dict)
    n_samples: int = 0
    ece_30d: dict[MarketType, float] = field(default_factory=dict)
    failure_flags: dict[MarketType, bool] = field(default_factory=dict)


def blend_probabilities(
    p_model: float,
    p_market: float,
    p_prior: float,
    weights: CalibrationWeights,
) -> float:
    """H.1 — p_blend = α·p_model + β·p_market + γ·p_prior"""
    p = weights.alpha * p_model + weights.beta * p_market + weights.gamma * p_prior
    return float(np.clip(p, 0.001, 0.999))


def calibrated_probability(
    p_blend: float,
    state: CalibrationState,
    market_type: MarketType = "moneyline",
) -> float:
    """
    H.4 — Final calibrated probability.
    Priority: isotonic → Platt → blend fallback.
    """
    if state.failure_flags.get(market_type, False):
        # Calibration failure: return blend with uncertainty penalty
        return float(0.9 * p_blend + 0.1 * 0.5)

    if state.isotonic_trained and state.isotonic is not None:
        try:
            p_iso = float(state.isotonic.predict([p_blend])[0])
            return float(np.clip(p_iso, 0.001, 0.999))
        except Exception:
            pass

    platt = state.platt.get(market_type)
    if platt and platt.trained:
        return float(np.clip(platt.predict(p_blend), 0.001, 0.999))

    # Fallback: uncertainty penalty
    return float(np.clip(0.9 * p_blend + 0.1 * 0.5, 0.001, 0.999))


def fit_isotonic(
    p_blends: np.ndarray,
    outcomes: np.ndarray,
) -> IsotonicRegression:
    """H.2 — Fit isotonic regression calibrator."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_blends, outcomes)
    return iso


def fit_platt(
    p_blends: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[float, float]:
    """H.3 — Fit Platt scaling via logistic regression with very high C."""
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=200)
    lr.fit(p_blends.reshape(-1, 1), outcomes)
    A = float(lr.coef_[0][0])
    B = float(lr.intercept_[0])
    return A, B


def compute_ece(
    p_preds: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    H.5 — Expected Calibration Error.
    ECE = Σ_b (|B_b|/N) × |acc(B_b) − conf(B_b)|
    Bin width: 1/n_bins.
    """
    N = len(outcomes)
    if N == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        mask = (p_preds >= lo) & (p_preds < hi)
        if not mask.any():
            continue
        acc = float(outcomes[mask].mean())
        conf = float(p_preds[mask].mean())
        ece += (mask.sum() / N) * abs(acc - conf)
    return ece


def brier_score(p_preds: np.ndarray, outcomes: np.ndarray) -> float:
    """H.6 — Brier score BS = (1/N) Σ (p_i − o_i)²"""
    return float(np.mean((p_preds - outcomes) ** 2))


def log_loss(p_preds: np.ndarray, outcomes: np.ndarray, eps: float = 1e-9) -> float:
    """H.6 — Log loss LL = −(1/N) Σ [o_i log p_i + (1−o_i) log(1−p_i)]"""
    p = np.clip(p_preds, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def combined_objective(p_preds: np.ndarray, outcomes: np.ndarray, lam: float = 0.3) -> float:
    """H.6 — L_total = BS + λ × LL"""
    return brier_score(p_preds, outcomes) + lam * log_loss(p_preds, outcomes)


def optimise_blend_weights(
    p_model_arr: np.ndarray,
    p_market_arr: np.ndarray,
    p_prior_arr: np.ndarray,
    outcomes: np.ndarray,
) -> CalibrationWeights:
    """
    H.1 — Optimise α, β, γ to minimise Brier score on validation set.
    Constraints: α + β + γ = 1, all ∈ [0, 1].
    Uses SLSQP.
    """
    def objective(w: np.ndarray) -> float:
        p = w[0] * p_model_arr + w[1] * p_market_arr + w[2] * p_prior_arr
        p = np.clip(p, 0.001, 0.999)
        return brier_score(p, outcomes)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * 3
    x0 = np.array([0.45, 0.40, 0.15])

    try:
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 500},
        )
        w = np.clip(res.x, 0.0, 1.0)
        # Renormalise
        w /= w.sum()
        return CalibrationWeights(alpha=float(w[0]), beta=float(w[1]), gamma=float(w[2]))
    except Exception as exc:
        logger.warning("Weight optimisation failed: %s", exc)
        return CalibrationWeights()


def generate_calibration_history(n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise historical (p_blend, outcome) pairs for demo calibration fitting.
    Introduces systematic overconfidence to make calibration non-trivial.
    """
    rng = np.random.default_rng(42)
    true_probs = rng.beta(5, 5, n)
    # Model is slightly overconfident
    p_blend = np.clip(true_probs + rng.normal(0, 0.05, n), 0.01, 0.99)
    outcomes = rng.binomial(1, true_probs).astype(float)
    return p_blend, outcomes


def build_calibration_state(
    p_blend_arr: np.ndarray,
    outcomes: np.ndarray,
    market_type: MarketType = "moneyline",
    weights: CalibrationWeights | None = None,
) -> CalibrationState:
    """Full calibration state build from training data."""
    state = CalibrationState()
    if weights:
        state.weights = weights
    state.n_samples = len(outcomes)

    if len(outcomes) >= MIN_CALIBRATION_SAMPLES:
        state.isotonic = fit_isotonic(p_blend_arr, outcomes)
        state.isotonic_trained = True
    else:
        # Platt only
        A, B = fit_platt(p_blend_arr, outcomes)
        state.platt[market_type] = PlattScaler(market_type=market_type, A=A, B=B, trained=True)

    p_cal = np.array([calibrated_probability(p, state, market_type) for p in p_blend_arr])
    ece = compute_ece(p_cal, outcomes)
    state.ece_30d[market_type] = ece
    state.failure_flags[market_type] = ece > 0.06

    return state
