"""NEXOM — Calibration Engine Page (Section H)."""
import streamlit as st
import numpy as np
import pandas as pd

from engines.calibration import (
    CalibrationWeights, CalibrationState,
    blend_probabilities, calibrated_probability,
    compute_ece, brier_score, log_loss, combined_objective,
    optimise_blend_weights, generate_calibration_history,
    build_calibration_state, MIN_CALIBRATION_SAMPLES,
)


def render():
    st.title("🔧 Calibration Engine (Silver Layer)")
    st.caption("Section H — Probability Blending, Isotonic Regression, Platt Scaling, ECE, Brier")

    tabs = st.tabs(["⚖️ Blending & Calibration", "📉 Calibration Curves", "🎯 ECE Monitor", "🏋️ Fit Calibrator"])

    # ── H.1 Blending ───────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("H.1 — Probability Blending")
        st.latex(r"p_{blend} = \alpha \cdot p_{model} + \beta \cdot p_{market} + \gamma \cdot p_{prior}")
        st.caption("α + β + γ = 1 | Typical: α≈0.45, β≈0.40, γ≈0.15")

        col1, col2, col3 = st.columns(3)
        with col1:
            p_model = st.slider("p_model (MC simulation)", 0.3, 0.9, 0.58, 0.01)
            alpha_w = st.slider("α (model weight)", 0.0, 1.0, 0.45, 0.01)
        with col2:
            p_market = st.slider("p_market (Shin-extracted)", 0.3, 0.9, 0.53, 0.01)
            beta_w = st.slider("β (market weight)", 0.0, 1.0, 0.40, 0.01)
        with col3:
            p_prior = st.slider("p_prior (Bayesian prior)", 0.3, 0.9, 0.50, 0.01)
            gamma_w_raw = max(0.0, 1.0 - alpha_w - beta_w)
            st.metric("γ (prior weight, auto)", f"{gamma_w_raw:.3f}")

        # Normalise weights
        total_w = alpha_w + beta_w + gamma_w_raw
        if total_w > 0:
            alpha_n = alpha_w / total_w
            beta_n = beta_w / total_w
            gamma_n = gamma_w_raw / total_w
        else:
            alpha_n, beta_n, gamma_n = 0.45, 0.40, 0.15

        weights = CalibrationWeights(alpha=alpha_n, beta=beta_n, gamma=gamma_n)
        p_blend = blend_probabilities(p_model, p_market, p_prior, weights)

        # Use stored calibration state if available
        if "cal_state" not in st.session_state:
            p_arr, outcomes = generate_calibration_history(600)
            st.session_state.cal_state = build_calibration_state(p_arr, outcomes, "moneyline", weights)

        state = st.session_state.cal_state
        p_final = calibrated_probability(p_blend, state, "moneyline")

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("p_blend", f"{p_blend:.4f}")
        bc2.metric("p_final (calibrated)", f"{p_final:.4f}")
        bc3.metric("Calibration method", "Isotonic" if state.isotonic_trained else "Platt/Fallback")
        bc4.metric("N training samples", state.n_samples)

        edge = p_final - p_market
        st.info(f"Edge after calibration: **{edge:+.4f}** ({edge*100:+.2f}pp vs market)")

        # H.4 priority display
        st.markdown("**H.4 — Final Probability Priority:**")
        st.code("""
p_final = p_iso           (if isotonic calibrator trained on ≥500 samples)
        = p_platt_m       (elif Platt calibrator trained for market type)
        = 0.9×p_blend + 0.1×0.5  (fallback with uncertainty penalty)
        """)

    # ── H.2/H.3 Calibration Curves ────────────────────────────────────────────
    with tabs[1]:
        st.subheader("H.2/H.3 — Calibration Curves (Reliability Diagram)")

        if "cal_state" not in st.session_state:
            p_arr, outcomes = generate_calibration_history(600)
            st.session_state.cal_state = build_calibration_state(p_arr, outcomes, "moneyline")

        state = st.session_state.cal_state
        p_arr, outcomes = generate_calibration_history(600)

        # Bin data for reliability diagram
        n_bins = 10
        bin_means_pred, bin_means_actual, bin_sizes = [], [], []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (p_arr >= lo) & (p_arr < hi)
            if mask.sum() > 0:
                bin_means_pred.append(float(p_arr[mask].mean()))
                bin_means_actual.append(float(outcomes[mask].mean()))
                bin_sizes.append(int(mask.sum()))

        if state.isotonic and state.isotonic_trained:
            p_cal = state.isotonic.predict(p_arr)
        else:
            p_cal = 0.9 * p_arr + 0.1 * 0.5

        bin_means_cal = []
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = (p_arr >= lo) & (p_arr < hi)
            if mask.sum() > 0:
                bin_means_cal.append(float(p_cal[mask].mean()))
            else:
                bin_means_cal.append(bin_means_pred[i] if i < len(bin_means_pred) else 0.0)

        min_len = min(len(bin_means_pred), len(bin_means_actual), len(bin_means_cal))
        rel_df = pd.DataFrame({
            "Mean Predicted": bin_means_pred[:min_len],
            "Empirical Accuracy (actual)": bin_means_actual[:min_len],
            "Post-Calibration": bin_means_cal[:min_len],
            "Perfect Calibration": bin_means_pred[:min_len],
        }).set_index("Mean Predicted")

        st.line_chart(rel_df[["Empirical Accuracy (actual)", "Post-Calibration", "Perfect Calibration"]])
        st.caption("Reliability diagram: Post-calibration should be close to the perfect diagonal.")

        # Metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Brier Score (raw)", f"{brier_score(p_arr, outcomes):.4f}")
        mc2.metric("Brier Score (calibrated)", f"{brier_score(p_cal, outcomes):.4f}")
        mc3.metric("ECE (calibrated)", f"{compute_ece(p_cal, outcomes):.4f}")

    # ── H.5 ECE Monitor ───────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("H.5 — Expected Calibration Error Monitor")
        st.latex(r"\text{ECE} = \sum_b \frac{|B_b|}{N} \cdot |\text{acc}(B_b) - \text{conf}(B_b)|")

        market_types = ["moneyline", "spread", "total", "player_prop"]
        np.random.seed(20)
        ece_data = {}
        for mt in market_types:
            ece_data[mt] = float(np.random.uniform(0.020, 0.065))

        st.session_state.calibration_failure_flags = {}
        rows = []
        for mt, ece_val in ece_data.items():
            failure = ece_val > 0.060
            st.session_state.calibration_failure_flags[mt] = failure
            rows.append({
                "Market Type": mt.title(),
                "ECE 30d": f"{ece_val:.4f}",
                "Target": "< 0.060",
                "Status": "✅ OK" if not failure else "❌ CALIBRATION_FAILURE",
                "Action": "Normal" if not failure else "Halt + wait for nightly refit",
            })

        df_ece = pd.DataFrame(rows)
        st.dataframe(df_ece, hide_index=True, use_container_width=True)

        any_failure = any(st.session_state.calibration_failure_flags.values())
        if any_failure:
            failed_markets = [k for k, v in st.session_state.calibration_failure_flags.items() if v]
            st.error(f"⚠️ CALIBRATION_FAILURE active for: {failed_markets}. All signals for these market types halted.")
        else:
            st.success("✅ All market types within ECE tolerance")

        # Combined objective
        st.markdown("---")
        st.subheader("H.6 — Combined Brier + Log-Loss Objective")
        st.latex(r"L_{total} = BS + 0.3 \cdot LL")
        p_demo, o_demo = generate_calibration_history(200)
        bs_val = brier_score(p_demo, o_demo)
        ll_val = log_loss(p_demo, o_demo)
        l_total = combined_objective(p_demo, o_demo, lam=0.3)
        c1, c2, c3 = st.columns(3)
        c1.metric("Brier Score", f"{bs_val:.4f}")
        c2.metric("Log Loss", f"{ll_val:.4f}")
        c3.metric("L_total (λ=0.3)", f"{l_total:.4f}")

    # ── Fit Calibrator ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Fit / Retrain Calibration State")

        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Training sample size", 100, 2000, 600, 50)
            market_type_sel = st.selectbox("Market type", ["moneyline", "spread", "total", "player_prop"])

        with col2:
            st.markdown("**Weight optimisation**")
            run_weight_opt = st.checkbox("Run SLSQP weight optimisation")

        if st.button("🔄 Fit Calibration State", type="primary"):
            with st.spinner("Fitting isotonic regression calibrator..."):
                p_arr, outcomes = generate_calibration_history(n_samples)

                if run_weight_opt and n_samples >= 100:
                    np.random.seed(42)
                    p_model_arr = np.clip(p_arr + np.random.normal(0, 0.03, n_samples), 0.01, 0.99)
                    p_market_arr = np.clip(p_arr - np.random.normal(0, 0.02, n_samples), 0.01, 0.99)
                    p_prior_arr = np.full(n_samples, 0.5)
                    weights = optimise_blend_weights(p_model_arr, p_market_arr, p_prior_arr, outcomes)
                    st.info(f"Optimised weights — α={weights.alpha:.3f}, β={weights.beta:.3f}, γ={weights.gamma:.3f}")
                else:
                    weights = None

                state = build_calibration_state(p_arr, outcomes, market_type_sel, weights)
                st.session_state.cal_state = state

            st.success(f"✅ Calibration state built | n={state.n_samples} | isotonic={'Yes' if state.isotonic_trained else 'No'}")
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("ECE", f"{state.ece_30d.get(market_type_sel, 0):.4f}")
            ec2.metric("Failure flag", str(state.failure_flags.get(market_type_sel, False)))
            ec3.metric("Min samples needed", MIN_CALIBRATION_SAMPLES)

            with st.expander("View iteration log"):
                st.json({
                    "n_samples": state.n_samples,
                    "isotonic_trained": state.isotonic_trained,
                    "weights": {"alpha": state.weights.alpha, "beta": state.weights.beta, "gamma": state.weights.gamma},
                    "ece": state.ece_30d,
                    "failure_flags": state.failure_flags,
                })
