"""NEXOM — Execution Microstructure Engine Page (Section G)."""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from engines.execution import (
    ExecutionParams, compute_ev_exec, compute_edge_net,
    compute_liquidity_factor, compute_timing_factor,
    compute_fill_probability, sharp_soft_divergence,
    execution_window_status, kelly_fraction, fractional_kelly_stake,
)


def render():
    st.title("⚡ Execution Microstructure Engine")
    st.caption("Section G — Bloom Layer: EV_exec, Latency Model, Sharp/Soft Divergence, Timing")

    tabs = st.tabs(["📐 EV_exec Calculator", "⏱️ Latency Model", "🔀 Sharp/Soft Signal", "🕐 Timing Windows"])

    # ── G.1 EV_exec ────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("G.1 — Execution Value Formula")
        st.latex(r"\text{EV}_{exec} = \text{edge}_{net} \times \text{liquidity} \times \text{timing} \times \text{fill\_prob} \times \text{latency\_adj}")

        col1, col2 = st.columns(2)
        with col1:
            p_final = st.slider("p_final (calibrated model prob)", 0.40, 0.80, 0.58, 0.01)
            p_market = st.slider("p_market (Shin-extracted)", 0.40, 0.70, 0.52, 0.01)
            intended_stake = st.number_input("Intended stake ($)", 10.0, 5000.0, 200.0, 10.0)
            estimated_liquidity = st.number_input("Est. liquidity at odds ($)", 100.0, 100000.0, 8000.0, 100.0)

        with col2:
            t_delay = st.slider("Signal delay (minutes)", 0.0, 15.0, 1.5, 0.1)
            book_stake_limit = st.number_input("Book stake limit ($)", 100.0, 50000.0, 2000.0, 100.0)
            decimal_odds = st.number_input("Decimal odds", 1.5, 5.0, 1.91, 0.01)
            bookmaker = st.selectbox("Bookmaker", ["pinnacle", "circa", "bet365", "fanduel", "default"])

        if st.button("⚡ Compute EV_exec", type="primary"):
            params = ExecutionParams(
                p_final=p_final, p_market=p_market,
                intended_stake=intended_stake,
                estimated_liquidity=estimated_liquidity,
                t_delay_minutes=t_delay,
                book_stake_limit=book_stake_limit,
                decimal_odds=decimal_odds,
                bookmaker=bookmaker,
            )
            result = compute_ev_exec(params)

            # Component breakdown
            st.markdown("### EV_exec Component Breakdown")
            comp_data = {
                "Component": ["edge_net", "liquidity_factor", "timing_factor", "fill_probability", "latency_adjustment", "**EV_exec**"],
                "Value": [
                    f"{result['edge_net']:.4f}",
                    f"{result['liquidity_factor']:.4f}",
                    f"{result['timing_factor']:.4f}",
                    f"{result['fill_probability']:.4f}",
                    f"{result['latency_adjustment']:.4f}",
                    f"**{result['ev_exec']:.4f}**",
                ],
                "Status": [
                    "✅" if result['edge_net'] > 0.030 else "⚠️ < 0.030",
                    "✅" if result['liquidity_factor'] > 0.50 else "⚠️ < 0.50",
                    "✅" if result['timing_factor'] > 0.35 else "⚠️ < 0.35",
                    "✅" if result['fill_probability'] > 0.50 else "⚠️ < 0.50",
                    "✅" if result['latency_adjustment'] > 0.85 else "⚠️",
                    "✅ POSITIVE" if result['ev_exec'] > 0 else "❌ NEGATIVE",
                ],
            }
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(df_comp, hide_index=True, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("EV_exec", f"{result['ev_exec']:.4f}",
                      delta_color="normal" if result['ev_exec'] > 0 else "inverse")
            c2.metric("Latency sampled", f"{result['latency_ms']:.0f}ms")
            c3.metric("Gate 1 (EV>0)", "✅ PASS" if result['ev_exec'] > 0 else "❌ FAIL")

            # Kelly stake
            f_star = kelly_fraction(p_final, decimal_odds)
            frac_stake = fractional_kelly_stake(p_final, decimal_odds, st.session_state.bankroll,
                                                st.session_state.kelly_fraction_multiplier)
            st.markdown("---")
            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("f* (full Kelly)", f"{f_star:.4f}")
            kc2.metric("Kelly multiplier", f"{st.session_state.kelly_fraction_multiplier:.0%}")
            kc3.metric("Recommended stake", f"${frac_stake:.2f}")

    # ── G.2 Latency Model ──────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("G.2 — Latency Model: LogNormal(μ_b, σ_b²)")
        st.latex(r"\text{latency}_b \sim \text{LogNormal}(\mu_b, \sigma_b^2)")
        st.latex(r"\text{odds\_received} = \text{odds\_offered} - \text{drift\_rate}_b \times \text{latency} / 1000")

        latency_params = {
            "pinnacle": {"mu": 5.5, "sigma": 0.4},
            "circa": {"mu": 5.5, "sigma": 0.4},
            "bet365": {"mu": 5.8, "sigma": 0.5},
            "fanduel": {"mu": 5.8, "sigma": 0.6},
            "caesars": {"mu": 5.9, "sigma": 0.6},
        }

        book_sel = st.selectbox("Select bookmaker for latency profile", list(latency_params.keys()))
        mu_b = latency_params[book_sel]["mu"]
        sigma_b = latency_params[book_sel]["sigma"]
        drift_rate = st.slider("Drift rate (decimal odds/s)", 0.0, 0.01, 0.002, 0.0005, format="%.4f")
        odds_offered = st.number_input("Odds offered", 1.5, 3.0, 1.92, 0.01)

        n_lat_samples = 10000
        lat_samples = np.random.lognormal(mu_b, sigma_b, n_lat_samples)
        odds_received = odds_offered - drift_rate * lat_samples / 1000

        lat_df = pd.DataFrame({
            "Latency (ms)": np.round(lat_samples[:200]).tolist(),
        })

        col_lat, col_stats = st.columns([3, 1])
        with col_lat:
            bins = np.linspace(0, np.percentile(lat_samples, 99), 50)
            hist, edges = np.histogram(lat_samples, bins=bins)
            hist_df = pd.DataFrame({
                "Latency (ms)": [(edges[i] + edges[i+1])/2 for i in range(len(hist))],
                "Frequency": hist.tolist(),
            })
            st.bar_chart(hist_df.set_index("Latency (ms)"))

        with col_stats:
            st.metric("μ_b", f"{mu_b}")
            st.metric("σ_b", f"{sigma_b}")
            st.metric("Median (ms)", f"{np.median(lat_samples):.0f}")
            st.metric("P95 (ms)", f"{np.percentile(lat_samples, 95):.0f}")
            lat_adj_mean = float(np.mean(np.maximum(0, (odds_received - 1) / (odds_offered - 1 + 1e-9))))
            st.metric("Mean latency_adj", f"{lat_adj_mean:.4f}")

    # ── G.3 Sharp/Soft Divergence ──────────────────────────────────────────────
    with tabs[2]:
        st.subheader("G.3 — Sharp/Soft Divergence Signal")
        st.latex(r"\Delta_{sharp-soft} = p_{sharp} - \overline{p_{soft}}")

        col1, col2 = st.columns(2)
        with col1:
            p_sharp = st.slider("p_sharp (Pinnacle Shin-prob)", 0.40, 0.70, 0.55, 0.01)
            st.markdown("**Soft book probabilities (Tier-3)**")
            p_soft_1 = st.slider("Soft book 1", 0.40, 0.70, 0.51, 0.01, key="ss1")
            p_soft_2 = st.slider("Soft book 2", 0.40, 0.70, 0.50, 0.01, key="ss2")
            p_soft_3 = st.slider("Soft book 3", 0.40, 0.70, 0.52, 0.01, key="ss3")

        result_ss = sharp_soft_divergence(p_sharp, [p_soft_1, p_soft_2, p_soft_3])

        with col2:
            st.markdown("**Signal Output**")
            delta = result_ss["delta"]
            signal = result_ss["signal"]
            modifier = result_ss["size_modifier"]

            color = "🟢" if signal == "sharp_corroborating" else ("🔴" if signal == "sharp_counter" else "🟡")
            st.metric("Δ_{sharp-soft}", f"{delta:+.4f}")
            st.metric(f"{color} Signal", signal.replace("_", " ").title())
            st.metric("Position size modifier", f"{modifier:.1f}×")

            if signal == "sharp_corroborating":
                st.success("Δ > 0.04: Sharp money corroborates model. Increase position by 1.2×")
            elif signal == "sharp_counter":
                st.error("Δ < −0.04: Sharp money against position. Reduce size by 0.7× regardless of model EV")
            else:
                st.info("|Δ| < 0.02: No clear divergence. Neutral positioning.")

    # ── G.4 Timing Windows ─────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("G.4 — Execution Timing Strategy")

        st.markdown("""
**Pre-game windows:**
- 🟢 **Primary:** T−90min → T−30min (liquid, not yet steam-prone)
- 🟡 **Secondary:** T−10min → T−5min (fresh injury info priced)
- 🔴 **Blackout:** T−2min → tip-off (spreads widen, fill prob < 0.4)

**Live:** Execute within 12 seconds of signal (timing_factor > 0.38)
        """)

        tip_offset = st.slider("Minutes until tip-off (T−X)", -10, 120, 45)
        tip_off = datetime.utcnow() + timedelta(minutes=tip_offset)
        window = execution_window_status(tip_off, datetime.utcnow())

        status_color = "✅" if window["valid"] else "❌"
        st.metric(f"{status_color} Window Status", window.get("window", "unknown").replace("_", " ").title())
        st.metric("Reason", window.get("reason", ""))
        if "minutes_to_tip" in window:
            st.metric("Minutes to tip", f"{window['minutes_to_tip']:.1f}")

        # Timing factor curve
        st.markdown("---")
        st.subheader("Timing Factor Decay: timing_factor = exp(−0.08 × t)")
        t_range = np.arange(0, 20, 0.1)
        tf_range = np.exp(-0.08 * t_range)
        df_tf = pd.DataFrame({"Minutes since signal": t_range, "timing_factor": tf_range})
        st.line_chart(df_tf.set_index("Minutes since signal"))
        st.caption("Threshold: timing_factor > 0.35 (≈ 13 minutes). Live threshold > 0.38 (≈ 12 seconds).")
