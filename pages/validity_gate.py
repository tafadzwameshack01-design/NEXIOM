"""NEXOM — Bet Validity Gate Page (Section L)."""
import streamlit as st
import numpy as np
import pandas as pd

from engines.validity_gate import GateInput, evaluate_gates, MIN_EV_BY_REGIME


def render():
    st.title("✅ Final Bet Validity Gate")
    st.caption("Section L — All 10 gates must pass simultaneously. Any single False → reject + log.")

    tabs = st.tabs(["🚦 Gate Evaluator", "📊 Gate Analytics", "📋 Gate Reference"])

    # ── Gate Evaluator ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("L — 10-Gate Simultaneous Evaluation")
        st.info("A bet is placed if and only if ALL 10 conditions evaluate to True simultaneously.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Probability & Edge (G1-G3)**")
            ev_exec = st.number_input("EV_exec", -0.10, 0.20, 0.042, 0.005, format="%.3f")
            p_final = st.slider("p_final (calibrated)", 0.30, 0.80, 0.58, 0.01)
            p_market = st.slider("p_market (Shin-extracted)", 0.30, 0.80, 0.53, 0.01)
            ece_30d = st.slider("ECE 30d", 0.01, 0.10, 0.035, 0.005)
            n_cal_samples = st.number_input("Calibration sample count", 0, 2000, 650, 50)
            cal_failure = st.checkbox("Calibration failure flag active", value=False)

        with col2:
            st.markdown("**Execution & Risk (G4-G6)**")
            liquidity_factor = st.slider("Liquidity factor", 0.0, 1.0, 0.72, 0.01)
            portfolio_ok = st.checkbox("Portfolio constraint passed", value=True)
            staleness = st.checkbox("Staleness flag", value=False)
            cal_fail_active = st.checkbox("Calibration failure (G6 flag)", value=False)
            vol_spike = st.checkbox("Volatility spike active", value=False)
            cat_stop = st.checkbox("Category stop active", value=False)
            regime_halt = st.checkbox("Regime halt active", value=False)
            j_stop = st.checkbox("Any J-section stop active", value=False)

        with col3:
            st.markdown("**Steam, Timing & Regime (G7-G10)**")
            steam_active = st.checkbox("Steam event active", value=False)
            steam_seconds = st.number_input("Steam detected (seconds ago)", 0, 600, 0, 10)
            timing_factor = st.slider("Timing factor", 0.0, 1.0, 0.78, 0.01)
            regime_sel = st.selectbox("Regime", ["REGIME_NORMAL", "REGIME_BULL", "REGIME_CAUTIOUS", "REGIME_HALT"])
            dc_shift = st.slider("DC M6 parameter shift", 0.0, 0.20, 0.04, 0.005)

        gate_input = GateInput(
            ev_exec=ev_exec,
            p_final=p_final,
            p_market=p_market,
            ece_30d=ece_30d,
            n_calibration_samples=n_cal_samples,
            calibration_failure_flag=cal_failure,
            liquidity_factor=liquidity_factor,
            portfolio_constraint_passed=portfolio_ok,
            staleness_flag=staleness,
            calibration_failure_active=cal_fail_active,
            volatility_spike_active=vol_spike,
            category_stop_active=cat_stop,
            regime_halt_active=regime_halt,
            any_j_stop_active=j_stop,
            steam_active=steam_active,
            steam_detected_seconds_ago=float(steam_seconds),
            timing_factor=timing_factor,
            regime=regime_sel,
            dc_m6_shift=dc_shift,
        )

        if st.button("🔍 Evaluate All 10 Gates", type="primary", use_container_width=True):
            result = evaluate_gates(gate_input)

            if result.passed:
                st.success("✅ ALL 10 GATES PASSED — Bet is eligible for placement")
            else:
                st.error(f"❌ {len(result.failed_gates)} GATE(S) FAILED — Bet REJECTED")
                st.write("**Failed gates:**", result.failed_gates)

            # Gate results table
            rows = []
            for gate_name, details in result.gate_details.items():
                rows.append({
                    "Gate": gate_name,
                    "Result": "✅ PASS" if details["passed"] else "❌ FAIL",
                    "Description": details.get("description", ""),
                })
            df_gates = pd.DataFrame(rows)
            st.dataframe(
                df_gates.style.apply(
                    lambda col: ["background-color: #1a3a1a" if "PASS" in str(v)
                                 else "background-color: #3a1a1a" for v in col]
                    if col.name == "Result" else [""] * len(col),
                    axis=0
                ),
                hide_index=True,
                use_container_width=True,
            )

            # Full JSON details in expander
            with st.expander("View full gate details (JSON)"):
                for gname, gdetails in result.gate_details.items():
                    st.json({gname: gdetails})

            # Log to rejected_bets if failed
            if not result.passed:
                st.session_state.rejected_bets.append({
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "failed_gates": result.failed_gates,
                    "p_final": p_final,
                    "p_market": p_market,
                    "ev_exec": ev_exec,
                })

    # ── Gate Analytics ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Gate Failure Analytics")

        n_rejected = len(st.session_state.rejected_bets)
        st.metric("Total rejected bets (session)", n_rejected)

        if st.session_state.rejected_bets:
            all_failed = []
            for b in st.session_state.rejected_bets:
                all_failed.extend(b.get("failed_gates", []))

            from collections import Counter
            counts = Counter(all_failed)
            if counts:
                freq_df = pd.DataFrame([
                    {"Gate": k, "Failure count": v, "% of rejections": f"{v/n_rejected*100:.1f}%"}
                    for k, v in sorted(counts.items(), key=lambda x: -x[1])
                ])
                st.dataframe(freq_df, hide_index=True, use_container_width=True)
                st.bar_chart(pd.DataFrame({"Count": dict(counts)}))
        else:
            st.info("No rejected bets in this session yet. Run gate evaluations above.")

        # Simulated historical gate failures
        st.markdown("---")
        st.subheader("Simulated Gate Failure Distribution (Historical)")
        gate_names = [f"GATE_{i}" for i in range(1, 11)]
        np.random.seed(99)
        failure_rates = np.array([0.08, 0.22, 0.05, 0.12, 0.07, 0.15, 0.18, 0.10, 0.20, 0.06])
        sim_df = pd.DataFrame({"Gate": gate_names, "Failure Rate": failure_rates})
        st.bar_chart(sim_df.set_index("Gate"))

    # ── Gate Reference ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Section L — Gate Formal Definitions")

        min_ev = MIN_EV_BY_REGIME.get(st.session_state.get("regime", "REGIME_NORMAL"), 0.030)
        gates_ref = [
            ("GATE_1_EV_POSITIVE", "EV_exec > 0.0", "Execution-adjusted EV must be strictly positive"),
            ("GATE_2_EDGE_MINIMUM", "(p_final − p_market) > 0.030", "Minimum 3.0 pp calibrated edge over market"),
            ("GATE_3_CALIBRATION_VALID", "ECE_30d < 0.060 AND n ≥ 500 AND no failure flag", "Calibration must be valid and well-trained"),
            ("GATE_4_LIQUIDITY_SUFFICIENT", "liquidity_factor > 0.50", "Intended stake < 50% of available liquidity"),
            ("GATE_5_PORTFOLIO_CONSTRAINT", "No SLSQP constraint violated", "Bet must fit within portfolio exposure limits"),
            ("GATE_6_NO_ACTIVE_FLAGS", "No active STALENESS, CALIBRATION_FAILURE, VOLATILITY_SPIKE, CATEGORY_STOP, REGIME_HALT, J-stop", "System must be in clean operational state"),
            ("GATE_7_STEAM_CLEAR", "No steam < 300s, OR EV_exec > 0.055 during steam", "Steam protection with elevated threshold"),
            ("GATE_8_TIMING_VALID", "timing_factor > 0.35", "Bet must be within viable execution window"),
            ("GATE_9_EXECUTION_VALUE", f"EV_exec > {min_ev:.3f} (in {st.session_state.get('regime', 'REGIME_NORMAL')})", "Regime-conditional minimum EV threshold"),
            ("GATE_10_DC_PARAMETER_STABLE", "M6 DC shift < 0.08", "Dixon-Coles parameters must not be drifting"),
        ]

        for gate_name, condition, description in gates_ref:
            with st.expander(f"**{gate_name}**"):
                st.markdown(f"**Condition:** `{condition}`")
                st.write(description)
