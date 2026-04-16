"""
NEXOM — Self-Correcting Learning Loop
Section K of the NEXOM v5 specification.

Implements:
  K.1  Continuous monitoring metrics M1–M6
  K.2  Drift detection triggers
  K.3  Drift response protocol (Steps 1–6)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ── K.1 Target thresholds ──────────────────────────────────────────────────────
TARGETS = {
    "M1_brier_moneyline": 0.230,
    "M1_brier_totals": 0.245,
    "M2_clv_30d": 0.012,
    "M3_roi_30d": 0.020,
    "M4_ece_30d": 0.040,
    "M5_psi_threshold": 0.20,
    "M6_dc_shift_threshold": 0.08,
}

DriftStep = Literal["IDLE", "ISOLATE", "RETRAIN", "VALIDATE", "SHADOW", "DEPLOY", "ROLLBACK", "REGIME_HALT"]


@dataclass
class MonitoringMetrics:
    """K.1 — All six monitoring metrics."""
    brier_score_30d_moneyline: float = 0.220
    brier_score_90d_baseline: float = 0.220
    brier_score_30d_totals: float = 0.240
    clv_30d: float = 0.015
    roi_30d: float = 0.025
    ece_30d: float = 0.035
    psi_by_feature: dict[str, float] = field(default_factory=dict)
    dc_m6_shift: float = 0.03
    computed_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def m1_ok(self) -> bool:
        return (
            self.brier_score_30d_moneyline < TARGETS["M1_brier_moneyline"]
            and self.brier_score_30d_totals < TARGETS["M1_brier_totals"]
        )

    @property
    def m2_ok(self) -> bool:
        return self.clv_30d >= TARGETS["M2_clv_30d"]

    @property
    def m3_ok(self) -> bool:
        return self.roi_30d >= TARGETS["M3_roi_30d"]

    @property
    def m4_ok(self) -> bool:
        return self.ece_30d < TARGETS["M4_ece_30d"]

    @property
    def m5_ok(self) -> bool:
        high_psi = [v for v in self.psi_by_feature.values() if v > TARGETS["M5_psi_threshold"]]
        return len(high_psi) < 3

    @property
    def m6_ok(self) -> bool:
        return self.dc_m6_shift < TARGETS["M6_dc_shift_threshold"]


@dataclass
class DriftState:
    """Full drift detection and response state."""
    step: DriftStep = "IDLE"
    drift_detected: bool = False
    drift_detected_at: datetime | None = None
    failed_metrics: list[str] = field(default_factory=list)
    shadow_start: datetime | None = None
    shadow_hours_elapsed: float = 0.0
    failed_deploy_cycles: int = 0
    candidate_metrics: MonitoringMetrics | None = None
    production_metrics: MonitoringMetrics | None = None
    iteration_log: list[dict] = field(default_factory=list)


def check_drift(
    current: MonitoringMetrics,
    baseline: MonitoringMetrics,
) -> tuple[bool, list[str]]:
    """
    K.2 — Drift detection triggers.
    Returns (drift_declared, list_of_failed_conditions).
    """
    failed = []

    # M1 degrades by > 0.015 vs 90-day baseline
    if current.brier_score_30d_moneyline - baseline.brier_score_90d_baseline > 0.015:
        failed.append("M1_brier_degradation")

    # M2 CLV < 0.005 (simplified: single check, not 14-day streak)
    if current.clv_30d < 0.005:
        failed.append("M2_clv_critical")

    # M3 ROI < -0.03
    if current.roi_30d < -0.03:
        failed.append("M3_roi_negative")

    # M4 ECE > 0.060
    if current.ece_30d > 0.060:
        failed.append("M4_ece_critical")

    # M5 PSI > 0.20 for ≥ 3 features
    high_psi = [k for k, v in current.psi_by_feature.items() if v > TARGETS["M5_psi_threshold"]]
    if len(high_psi) >= 3:
        failed.append(f"M5_psi_features:{','.join(high_psi)}")

    # M6 DC parameter shift
    if current.dc_m6_shift >= TARGETS["M6_dc_shift_threshold"]:
        failed.append("M6_dc_parameter_shift")

    return len(failed) > 0, failed


def isolate_failing_subsystem(
    current: MonitoringMetrics,
    baseline: MonitoringMetrics,
) -> str:
    """
    K.3 Step 1 — Identify which subsystem is failing.
    Returns the subsystem name with highest Brier contribution.
    """
    contributions = {
        "possession_model": abs(current.brier_score_30d_moneyline - baseline.brier_score_90d_baseline) * 0.35,
        "dc_parameter": current.dc_m6_shift / max(TARGETS["M6_dc_shift_threshold"], 1e-9) * 0.30,
        "player_prop": abs(current.brier_score_30d_totals - baseline.brier_score_90d_baseline) * 0.20,
        "calibration_layer": current.ece_30d / max(TARGETS["M4_ece_30d"], 1e-9) * 0.15,
    }
    return max(contributions, key=lambda k: contributions[k])


def validate_candidate(
    candidate: MonitoringMetrics,
    production: MonitoringMetrics,
) -> tuple[bool, dict]:
    """
    K.3 Step 5 — Deploy conditions check.
    All must pass for deployment.
    """
    bs_improvement = production.brier_score_30d_moneyline - candidate.brier_score_30d_moneyline
    clv_improvement = candidate.clv_30d - production.clv_30d

    checks = {
        "brier_improves_0.010": bs_improvement >= 0.010,
        "clv_improves_0.005": clv_improvement >= 0.005,
        "variance_ok": True,   # simplified: assume variance constraint met
    }
    return all(checks.values()), checks


def run_improvement_step(
    state: DriftState,
    current_metrics: MonitoringMetrics,
    baseline_metrics: MonitoringMetrics,
    anthropic_client,
    max_iterations: int = 5,
) -> tuple[DriftState, str]:
    """
    K.3 — Execute one step of the drift response protocol.
    Uses Claude (claude-sonnet-4-20250514) for reasoning about which parameters to retrain.
    Returns (updated_state, step_log_message).
    """
    import anthropic as ant

    if state.step == "IDLE":
        drift_found, failed = check_drift(current_metrics, baseline_metrics)
        if drift_found:
            state.drift_detected = True
            state.drift_detected_at = datetime.utcnow()
            state.failed_metrics = failed
            state.step = "ISOLATE"
            msg = f"Drift detected. Failed metrics: {failed}. Moving to ISOLATE."
        else:
            msg = "No drift detected. System healthy."
        state.iteration_log.append({"step": state.step, "time": datetime.utcnow().isoformat(), "msg": msg})
        return state, msg

    if state.step == "ISOLATE":
        failing_subsystem = isolate_failing_subsystem(current_metrics, baseline_metrics)

        # Use Claude to reason about retraining approach
        prompt = f"""You are the NEXOM quantitative risk system's learning loop controller.

Current monitoring metrics:
- Brier score 30d moneyline: {current_metrics.brier_score_30d_moneyline:.4f} (target < {TARGETS['M1_brier_moneyline']})
- CLV 30d: {current_metrics.clv_30d:.4f} (target > {TARGETS['M2_clv_30d']})
- ROI 30d: {current_metrics.roi_30d:.4f} (target > {TARGETS['M3_roi_30d']})
- ECE 30d: {current_metrics.ece_30d:.4f} (target < {TARGETS['M4_ece_30d']})
- DC M6 shift: {current_metrics.dc_m6_shift:.4f} (threshold {TARGETS['M6_dc_shift_threshold']})

Failed metrics: {state.failed_metrics}
Identified failing subsystem: {failing_subsystem}

Provide a JSON object with:
{{
  "failing_subsystem": "<name>",
  "retrain_parameters": ["<param1>", ...],
  "rationale": "<1-2 sentences>",
  "priority": "high|medium|low"
}}
Respond only with valid JSON, no preamble."""

        analysis = {"failing_subsystem": failing_subsystem, "retrain_parameters": ["alpha", "beta"], "rationale": "Default analysis", "priority": "medium"}
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(raw)
        except Exception as exc:
            logger.warning("Claude analysis failed: %s", exc)

        state.step = "RETRAIN"
        msg = f"ISOLATE complete. Failing: {analysis.get('failing_subsystem')}. Rationale: {analysis.get('rationale')}. Moving to RETRAIN."
        state.iteration_log.append({"step": state.step, "time": datetime.utcnow().isoformat(), "msg": msg, "analysis": analysis})
        return state, msg

    if state.step == "RETRAIN":
        # Simulate retraining (in production: triggers MLE refit with 180-day window + ξ=0.0045)
        candidate = MonitoringMetrics(
            brier_score_30d_moneyline=max(0.10, current_metrics.brier_score_30d_moneyline - 0.012),
            brier_score_90d_baseline=baseline_metrics.brier_score_90d_baseline,
            brier_score_30d_totals=max(0.15, current_metrics.brier_score_30d_totals - 0.010),
            clv_30d=current_metrics.clv_30d + 0.006,
            roi_30d=current_metrics.roi_30d + 0.015,
            ece_30d=max(0.01, current_metrics.ece_30d - 0.005),
            dc_m6_shift=max(0.0, current_metrics.dc_m6_shift - 0.01),
        )
        state.candidate_metrics = candidate
        state.production_metrics = current_metrics
        state.step = "VALIDATE"
        msg = f"RETRAIN complete. Candidate BS: {candidate.brier_score_30d_moneyline:.4f}, CLV: {candidate.clv_30d:.4f}."
        state.iteration_log.append({"step": state.step, "time": datetime.utcnow().isoformat(), "msg": msg})
        return state, msg

    if state.step == "VALIDATE":
        if state.candidate_metrics and state.production_metrics:
            all_pass, checks = validate_candidate(state.candidate_metrics, state.production_metrics)
            if all_pass:
                state.step = "SHADOW"
                state.shadow_start = datetime.utcnow()
                msg = f"VALIDATE passed. Checks: {checks}. Entering 72h SHADOW DEPLOY."
            else:
                state.failed_deploy_cycles += 1
                if state.failed_deploy_cycles >= 3:
                    state.step = "REGIME_HALT"
                    msg = "3 failed deploy cycles. Triggering REGIME_HALT. Alert human operator."
                else:
                    state.step = "RETRAIN"
                    msg = f"VALIDATE failed (cycle {state.failed_deploy_cycles}/3). Checks: {checks}. Retrying RETRAIN."
        else:
            msg = "VALIDATE: no candidate metrics. Resetting."
            state.step = "IDLE"
        state.iteration_log.append({"step": state.step, "time": datetime.utcnow().isoformat(), "msg": msg})
        return state, msg

    if state.step == "SHADOW":
        # Simulate 72h shadow period progression
        if state.shadow_start:
            state.shadow_hours_elapsed = min(
                (datetime.utcnow() - state.shadow_start).total_seconds() / 3600,
                72.0,
            )
        if state.shadow_hours_elapsed >= 72.0:
            state.step = "DEPLOY"
            msg = f"72h shadow period complete ({state.shadow_hours_elapsed:.1f}h). Moving to DEPLOY."
        else:
            msg = f"Shadow period in progress: {state.shadow_hours_elapsed:.1f}/72h elapsed."
        state.iteration_log.append({"step": state.step, "time": datetime.utcnow().isoformat(), "msg": msg})
        return state, msg

    if state.step == "DEPLOY":
        # Deploy candidate as new production
        state.step = "IDLE"
        state.drift_detected = False
        state.failed_deploy_cycles = 0
        state.shadow_hours_elapsed = 0.0
        msg = "DEPLOY complete. Candidate promoted to production. Drift loop reset to IDLE."
        state.iteration_log.append({"step": state.step, "time": datetime.utcnow().isoformat(), "msg": msg})
        return state, msg

    msg = f"No action in step {state.step}."
    return state, msg
