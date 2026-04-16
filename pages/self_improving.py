"""NEXOM — Self-Correcting Learning Loop Page (Section K)."""
import streamlit as st
import numpy as np
import pandas as pd
import os
import anthropic

from engines.self_improving import (
    MonitoringMetrics, DriftState, check_drift,
    run_improvement_step, TARGETS,
)


def render():
    st.title("🔄 Self-Correcting Learning Loop")
    st.caption("Section K — Drift Detection, Shadow Deploy, Rollback Protocol | Powered by Claude")

    tabs = st.tabs(["📊 Metrics Monitor", "🔍 Drift Detection", "⚙️ Improvement Loop", "📋 Protocol Reference"])

    # ── K.1 Metrics Monitor ────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("K.1 — Continuous Monitoring Metrics")

        col1, col2 = st.columns(2)
        with col1:
            bs_mono = st.slider("M1: Brier Score (moneyline 30d)", 0.15, 0.30, 0.218, 0.001, format="%.3f")
            bs_base = st.slider("M1: Brier Score (90d baseline)", 0.15, 0.30, 0.215, 0.001, format="%.3f")
            bs_totals = st.slider("M1: Brier Score (totals 30d)", 0.15, 0.30, 0.235, 0.001, format="%.3f")
            clv = st.slider("M2: CLV 30d", -0.02, 0.05, 0.013, 0.001, format="%.3f")
        with col2:
            roi = st.slider("M3: ROI 30d", -0.10, 0.15, 0.022, 0.001, format="%.3f")
            ece = st.slider("M4: ECE 30d", 0.01, 0.10, 0.038, 0.001, format="%.3f")
            dc_shift = st.slider("M6: DC parameter shift", 0.0, 0.15, 0.04, 0.005, format="%.3f")

        np.random.seed(8)
        psi_features = {f"feature_{i}": float(np.random.uniform(0.05, 0.35)) for i in range(6)}

        current = MonitoringMetrics(
            brier_score_30d_moneyline=bs_mono,
            brier_score_90d_baseline=bs_base,
            brier_score_30d_totals=bs_totals,
            clv_30d=clv,
            roi_30d=roi,
            ece_30d=ece,
            psi_by_feature=psi_features,
            dc_m6_shift=dc_shift,
        )

        metrics_rows = [
            {"Metric": "M1 Brier (moneyline)", "Value": f"{bs_mono:.4f}", "Target": f"< {TARGETS['M1_brier_moneyline']}", "Status": "✅" if current.m1_ok else "⚠️"},
            {"Metric": "M1 Brier (totals)", "Value": f"{bs_totals:.4f}", "Target": f"< {TARGETS['M1_brier_totals']}", "Status": "✅" if bs_totals < TARGETS["M1_brier_totals"] else "⚠️"},
            {"Metric": "M2 CLV 30d", "Value": f"{clv:.4f}", "Target": f"> {TARGETS['M2_clv_30d']}", "Status": "✅" if current.m2_ok else "⚠️"},
            {"Metric": "M3 ROI 30d", "Value": f"{roi:.4f}", "Target": f"> {TARGETS['M3_roi_30d']}", "Status": "✅" if current.m3_ok else "⚠️"},
            {"Metric": "M4 ECE 30d", "Value": f"{ece:.4f}", "Target": f"< {TARGETS['M4_ece_30d']}", "Status": "✅" if current.m4_ok else "⚠️"},
            {"Metric": "M5 PSI (>3 features > 0.20)", "Value": str(sum(1 for v in psi_features.values() if v > 0.20)), "Target": "< 3 features", "Status": "✅" if current.m5_ok else "⚠️"},
            {"Metric": "M6 DC shift", "Value": f"{dc_shift:.4f}", "Target": f"< {TARGETS['M6_dc_shift_threshold']}", "Status": "✅" if current.m6_ok else "⚠️"},
        ]

        df_metrics = pd.DataFrame(metrics_rows)
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

        all_ok = all([current.m1_ok, current.m2_ok, current.m3_ok, current.m4_ok, current.m5_ok, current.m6_ok])
        if all_ok:
            st.success("✅ All monitoring metrics within target thresholds")
        else:
            failed = [r["Metric"] for r in metrics_rows if r["Status"] == "⚠️"]
            st.warning(f"⚠️ Metrics outside targets: {failed}")

        st.session_state["current_metrics"] = current

    # ── K.2 Drift Detection ────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("K.2 — Drift Detection Triggers")
        st.markdown("""
Drift is declared if **ANY** of:
- M1 degrades by > 0.015 vs 90-day baseline
- M2 CLV drops below 0.005 for 14 consecutive days
- M3 ROI drops below −0.03 (30-day)
- M4 ECE exceeds 0.060
- M5 PSI > 0.20 for ≥ 3 features simultaneously
- M6 DC parameter shift exceeds threshold
        """)

        current = st.session_state.get("current_metrics", MonitoringMetrics())
        baseline = MonitoringMetrics(
            brier_score_90d_baseline=0.215,
            clv_30d=0.015, roi_30d=0.030, ece_30d=0.030,
        )

        if st.button("🔍 Run Drift Detection", type="primary"):
            drift_found, failed_conditions = check_drift(current, baseline)
            st.session_state.drift_status = {"active": drift_found, "failed": failed_conditions}

            if drift_found:
                st.error(f"🔴 DRIFT DECLARED — {len(failed_conditions)} condition(s) failed")
                for c in failed_conditions:
                    st.write(f"  • {c}")
                st.info("Protocol: ISOLATE → RETRAIN → VALIDATE → SHADOW (72h) → DEPLOY or ROLLBACK")
            else:
                st.success("✅ No drift detected. System operating within normal bounds.")

        if st.session_state.drift_status.get("active"):
            st.error(f"🔴 DRIFT ACTIVE | Failed: {st.session_state.drift_status.get('failed', [])}")

    # ── K.3 Improvement Loop ───────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("K.3 — Drift Response Protocol (Claude-Powered)")
        st.caption("Steps: IDLE → ISOLATE → RETRAIN → VALIDATE → SHADOW → DEPLOY (or ROLLBACK)")

        if "drift_state" not in st.session_state:
            st.session_state.drift_state = DriftState()

        drift_state: DriftState = st.session_state.drift_state

        # Status display
        step_colors = {
            "IDLE": "🔵", "ISOLATE": "🟡", "RETRAIN": "🟠",
            "VALIDATE": "🟣", "SHADOW": "🟤", "DEPLOY": "🟢",
            "ROLLBACK": "🔴", "REGIME_HALT": "⛔",
        }
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric(f"{step_colors.get(drift_state.step, '⚪')} Current Step", drift_state.step)
        sc2.metric("Drift Active", str(drift_state.drift_detected))
        sc3.metric("Failed Deploy Cycles", drift_state.failed_deploy_cycles)

        if drift_state.step == "SHADOW" and drift_state.shadow_start:
            st.progress(min(drift_state.shadow_hours_elapsed / 72.0, 1.0),
                        text=f"Shadow period: {drift_state.shadow_hours_elapsed:.1f}/72h")

        max_iter = st.slider("Max iterations", 1, 5, 5)
        progress_bar = st.empty()
        log_container = st.empty()

        col_run, col_stop, col_reset = st.columns(3)
        with col_run:
            run_loop = st.button("▶️ Run Improvement Step", type="primary")
        with col_stop:
            if st.button("⏹ Stop Loop"):
                st.session_state.stop_loop = True
        with col_reset:
            if st.button("🔁 Reset Loop"):
                st.session_state.drift_state = DriftState()
                st.rerun()

        if run_loop:
            st.session_state.stop_loop = False
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                st.error("ANTHROPIC_API_KEY not set in environment. Cannot run Claude-powered improvement loop.")
            else:
                client = anthropic.Anthropic(api_key=api_key)
                current = st.session_state.get("current_metrics", MonitoringMetrics(
                    brier_score_30d_moneyline=0.235,
                    brier_score_90d_baseline=0.215,
                    clv_30d=0.008,
                    roi_30d=-0.01,
                    ece_30d=0.055,
                    dc_m6_shift=0.07,
                ))
                baseline = MonitoringMetrics(brier_score_90d_baseline=0.215)

                for i in range(max_iter):
                    if st.session_state.stop_loop:
                        st.warning("⏹ Loop stopped by user")
                        break

                    progress_bar.progress((i + 1) / max_iter, text=f"Step {i+1}/{max_iter}: {drift_state.step}")

                    with st.spinner(f"Executing step: {drift_state.step}..."):
                        drift_state, step_msg = run_improvement_step(
                            drift_state, current, baseline, client, max_iter
                        )
                        st.session_state.improvement_iteration = i + 1

                    st.session_state.drift_state = drift_state

                    if drift_state.step in ("IDLE", "REGIME_HALT", "ROLLBACK"):
                        break

                progress_bar.progress(1.0, text="Complete")
                st.success(f"Loop completed. Final step: {drift_state.step}")

        # Iteration log
        if drift_state.iteration_log:
            with st.expander("📋 Iteration Log", expanded=True):
                for entry in reversed(drift_state.iteration_log[-10:]):
                    step = entry.get("step", "")
                    msg = entry.get("msg", "")
                    t = entry.get("time", "")
                    color = step_colors.get(step, "⚪")
                    st.write(f"**{color} [{t[:19]}] {step}:** {msg}")

    # ── Protocol Reference ─────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("K.3 — Protocol Reference")
        steps = [
            ("Step 1: ISOLATE", "Identify failing subsystem by decomposing Brier score into: possession model, DC parameter, player prop, and calibration layer components via held-out validation split."),
            ("Step 2: RETRAIN", "Retrain only the identified failing subsystem using most recent 180-day window with ξ=0.0045 temporal decay weighting. All other subsystems remain frozen."),
            ("Step 3: VALIDATE", "Run full simulation (N_pre=100,000) on held-out last-30-days test set. Compute all M1–M6 metrics for candidate new model."),
            ("Step 4: SHADOW DEPLOY", "Run candidate model alongside production for 72 hours. Both generate signals; only production signals are executed. Compare M1–M6 between candidate and production."),
            ("Step 5: DEPLOY", "Deploy if: candidate BS improves ≥ 0.010, CLV improves ≥ 0.005, variance increase ≤ 10%."),
            ("Step 6: ROLLBACK", "If deploy criteria not met after 72h shadow: retain production, extend shadow by 72h with expanded feature set. After 3 failed cycles → REGIME_HALT + alert human operator via Slack."),
        ]
        for title, desc in steps:
            st.markdown(f"**{title}**")
            st.write(desc)
            st.markdown("---")
