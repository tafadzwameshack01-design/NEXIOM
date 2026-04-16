"""NEXOM — Monte Carlo Engine Page (Section E)."""
import streamlit as st
import numpy as np
import pandas as pd
import time

from engines.monte_carlo import MCInput, run_simulation, get_score_distribution, N_PRE, N_LIVE


def render():
    st.title("🎯 Vectorised Monte Carlo Engine")
    st.caption("Section E — NumPy-only, Gumbel-max trick, N_pre=100,000 / N_live=25,000")

    tabs = st.tabs(["🚀 Run Simulation", "📊 Score Distributions", "⚡ Live Mode", "📐 Spec Reference"])

    with tabs[0]:
        st.subheader("E.3 — Full Pre-Game Simulation")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Home Team Parameters**")
            alpha_h = st.slider("α_home (offense mean)", 0.5, 2.0, 1.08, 0.01, key="mc_ah")
            alpha_h_std = st.slider("α_home σ (uncertainty)", 0.01, 0.2, 0.05, 0.01, key="mc_ahs")
            beta_h = st.slider("β_home (defense mean)", 0.5, 2.0, 0.97, 0.01, key="mc_bh")
            fatigue_h = st.slider("Fatigue index home", 0.0, 1.0, 0.1, 0.05, key="mc_fh")

        with col2:
            st.markdown("**Away Team Parameters**")
            alpha_a = st.slider("α_away (offense mean)", 0.5, 2.0, 0.95, 0.01, key="mc_aa")
            alpha_a_std = st.slider("α_away σ (uncertainty)", 0.01, 0.2, 0.05, 0.01, key="mc_aas")
            beta_a = st.slider("β_away (defense mean)", 0.5, 2.0, 1.03, 0.01, key="mc_ba")
            fatigue_a = st.slider("Fatigue index away", 0.0, 1.0, 0.15, 0.05, key="mc_fa")

        with col3:
            st.markdown("**Simulation Settings**")
            pace = st.slider("Pace (poss/48 min)", 80.0, 115.0, 98.0, 0.5, key="mc_pace")
            spread = st.slider("Spread line", -20.0, 20.0, -3.5, 0.5, key="mc_spread")
            total_line = st.slider("O/U total line", 190.0, 260.0, 220.5, 0.5, key="mc_total")
            n_mode = st.selectbox("N simulations", [1000, 5000, 10000, 25000, 100000],
                                   index=2, key="mc_n")
            gamma = st.slider("γ (home court)", 0.8, 1.2, 1.035, 0.005, key="mc_gamma")

        col_btn, col_stop = st.columns([3, 1])
        with col_btn:
            run_btn = st.button("▶️ Run Simulation", type="primary", use_container_width=True)
        with col_stop:
            if st.button("⏹ Stop", use_container_width=True):
                st.session_state.stop_loop = True

        if run_btn:
            st.session_state.stop_loop = False
            inp = MCInput(
                alpha_home_mean=alpha_h, alpha_away_mean=alpha_a,
                beta_home_mean=beta_h, beta_away_mean=beta_a,
                alpha_home_std=alpha_h_std, alpha_away_std=alpha_a_std,
                beta_home_std=0.05, beta_away_std=0.05,
                gamma=gamma, pace=pace, spread=spread,
                total_line=total_line, fatigue_home=fatigue_h,
                fatigue_away=fatigue_a, n_simulations=n_mode,
            )
            with st.spinner(f"Running {n_mode:,} simulations (vectorised NumPy)..."):
                out = run_simulation(inp)

            st.session_state.simulation_outputs["last"] = {
                "p_home_win": out.p_home_win,
                "p_cover_spread": out.p_cover_spread,
                "p_over_total": out.p_over_total,
                "p_overtime": out.p_overtime,
                "mean_total": out.mean_total,
                "std_total": out.std_total,
                "mean_margin": out.mean_margin,
                "std_margin": out.std_margin,
                "elapsed": out.elapsed_seconds,
                "n": n_mode,
            }

            timing_ok = (n_mode == N_PRE and out.elapsed_seconds <= 45) or \
                        (n_mode == N_LIVE and out.elapsed_seconds <= 8) or \
                        n_mode not in (N_PRE, N_LIVE)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("P(Home Win)", f"{out.p_home_win:.4f}", f"{out.p_home_win*100:.1f}%")
            c2.metric("P(Cover Spread)", f"{out.p_cover_spread:.4f}")
            c3.metric("P(Over Total)", f"{out.p_over_total:.4f}")
            c4.metric("P(Overtime)", f"{out.p_overtime:.4f}")
            c5.metric("P(Home Alt Cover)", f"{out.p_home_cover_alt:.4f}")

            st.markdown("---")
            c6, c7, c8, c9 = st.columns(4)
            c6.metric("Mean Total", f"{out.mean_total:.1f}")
            c7.metric("Std Total", f"{out.std_total:.1f}")
            c8.metric("Mean Margin", f"{out.mean_margin:+.1f}")
            c9.metric(
                "Elapsed",
                f"{out.elapsed_seconds:.2f}s",
                delta="✅ Within budget" if timing_ok else "⚠️ Over budget",
                delta_color="normal" if timing_ok else "inverse",
            )

            if not timing_ok:
                st.warning(f"E.4 LATENCY_FLAG: {n_mode:,} sims took {out.elapsed_seconds:.2f}s (budget: {'45s' if n_mode==N_PRE else '8s'}). Reduce N by 20%.")

    with tabs[1]:
        st.subheader("Score Distribution Analysis")

        if st.button("🔄 Generate Score Distributions", key="dist_btn"):
            inp = MCInput(
                alpha_home_mean=1.08, alpha_away_mean=0.95,
                beta_home_mean=0.97, beta_away_mean=1.03,
                pace=98.0, spread=-3.5, total_line=220.5,
                n_simulations=5000,
            )
            with st.spinner("Sampling score distributions..."):
                dist_data = get_score_distribution(inp, bins=40)

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Margin Distribution (Home − Away)**")
                df_margin = pd.DataFrame({
                    "Margin": dist_data["margin_bins"],
                    "Frequency": dist_data["margin_hist"],
                })
                st.bar_chart(df_margin.set_index("Margin"))

            with col_right:
                st.markdown("**Total Points Distribution**")
                df_total = pd.DataFrame({
                    "Total": dist_data["total_bins"],
                    "Frequency": dist_data["total_hist"],
                })
                st.bar_chart(df_total.set_index("Total"))

            # Scatter sample
            st.markdown("**Simulated Final Scores (sample of 500)**")
            df_scatter = pd.DataFrame({
                "Home Score": dist_data["score_home"][:500],
                "Away Score": dist_data["score_away"][:500],
            })
            st.scatter_chart(df_scatter, x="Home Score", y="Away Score")

    with tabs[2]:
        st.subheader("E.1 — Live Mode (N_live = 25,000)")
        st.info("Live mode simulates from current game state forward. Triggered on every canonical possession event.")

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            score_home = st.number_input("Current Score Home", 0, 160, 54, key="live_sh")
            score_away = st.number_input("Current Score Away", 0, 160, 51, key="live_sa")
        with lc2:
            period = st.selectbox("Period", [1, 2, 3, 4, 5], index=2, key="live_period")
            time_rem_min = st.slider("Minutes remaining in period", 0.0, 12.0, 6.5, 0.5)
        with lc3:
            pace_live = st.slider("Live pace", 80.0, 115.0, 96.0, 0.5, key="live_pace")
            total_line_live = st.slider("O/U line (live)", 190.0, 260.0, 219.5, 0.5, key="live_total")

        periods_remaining = max(0, 4 - period)
        time_remaining_total = periods_remaining * 720 + time_rem_min * 60

        if st.button("⚡ Run Live Simulation (N=25,000)", key="live_run"):
            inp_live = MCInput(
                alpha_home_mean=1.05, alpha_away_mean=0.98,
                beta_home_mean=0.99, beta_away_mean=1.01,
                pace=pace_live, spread=-3.5, total_line=total_line_live,
                score_home=score_home, score_away=score_away,
                time_remaining_total=time_remaining_total,
                n_simulations=N_LIVE,
            )
            with st.spinner("Live simulation running (25,000 paths)..."):
                out_live = run_simulation(inp_live)

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("P(Home Win)", f"{out_live.p_home_win:.4f}")
            rc2.metric("P(Over)", f"{out_live.p_over_total:.4f}")
            rc3.metric("Elapsed", f"{out_live.elapsed_seconds:.2f}s",
                       delta="✅ ≤ 8s" if out_live.elapsed_seconds <= 8 else "⚠️ > 8s",
                       delta_color="normal" if out_live.elapsed_seconds <= 8 else "inverse")
            rc4.metric("Current Margin", f"{score_home - score_away:+d}")

    with tabs[3]:
        st.subheader("Section E — Specification Reference")
        st.markdown("""
**E.2 Vectorisation Requirements:**
- ❌ No Python loop over N simulations
- ✅ All N-axis ops via NumPy array operations
- ✅ Gumbel-max trick: `argmax(Gumbel(0,1,(N,8)) + log(probs))` for vectorised multinomial sampling
- ✅ Single Python loop over possession steps only (max 250)

**E.3 Simulation Steps:**
1. Sample α_h, α_a, β_h, β_a from posterior Normal distributions — shape (N,)
2. Sample pace from Normal(π_g, σ_pace=3.5) clipped to [80, 115]
3. Per possession: vectorised multinomial via Gumbel-max → points lookup → accumulate
4. OT extension for tied simulations (max 3 OT periods)
5. Compute P(home_win), P(cover), P(over), P(OT) via np.mean

**E.4 Timing Constraints:**
- N_pre=100,000 → ≤ 45 seconds
- N_live=25,000 → ≤ 8 seconds
- On breach: reduce N by 20%, set LATENCY_FLAG
        """)
