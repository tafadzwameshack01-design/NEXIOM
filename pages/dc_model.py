"""NEXOM — Dixon–Coles Model Page (Section B)."""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from engines.dixon_coles import (
    fit_dc_parameters, generate_synthetic_games, incremental_newton_update,
    parameter_stability_score, temporal_weight, DCParameters, GameRecord,
    XI, GAMMA_INIT, L_AVG_DEFAULT,
)


def render():
    st.title("📊 Dixon–Coles Basketball Adaptation")
    st.caption("Section B — Attack/Defense MLE, Temporal Decay, Bivariate Correction")

    tabs = st.tabs(["⚙️ Parameter Fitting", "📉 Temporal Decay", "🔬 Parameter Explorer", "📋 Game Records"])

    # ── Tab 1: Parameter Fitting ───────────────────────────────────────────────
    with tabs[0]:
        st.subheader("B.3 — Maximum Likelihood Estimation (L-BFGS-B)")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_teams = st.slider("Number of Teams", 4, 30, 10)
            n_games = st.slider("Number of Historical Games", 50, 500, 200)
        with c2:
            gamma_init = st.number_input("γ initial (home court)", 0.8, 1.2, float(GAMMA_INIT), 0.005)
            xi_val = st.number_input("ξ decay constant (days⁻¹)", 0.001, 0.01, float(XI), 0.0005, format="%.4f")
        with c3:
            max_iter = st.slider("L-BFGS-B max iterations", 50, 500, 200)
            l_avg = st.number_input("L_avg (pts/100 poss)", 100.0, 125.0, float(L_AVG_DEFAULT))

        if st.button("🔄 Fit DC Parameters", type="primary"):
            with st.spinner("Running L-BFGS-B optimisation..."):
                games, team_ids = generate_synthetic_games(n_teams, n_games)
                dc = fit_dc_parameters(
                    games,
                    current_date=datetime.utcnow(),
                    l_avg=l_avg,
                    max_iter=max_iter,
                )
                st.session_state.dc_parameters = {
                    "alpha": dc.alpha, "beta": dc.beta,
                    "gamma": dc.gamma, "phi_h": dc.phi_h, "phi_a": dc.phi_a,
                    "rho": dc.rho, "n_games": dc.n_games, "l_avg": dc.l_avg,
                }

            st.success(f"✅ Fitted on {n_games} games across {n_teams} teams")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("γ (home advantage)", f"{dc.gamma:.4f}")
            col2.metric("φ_h (overdispersion)", f"{dc.phi_h:.3f}")
            col3.metric("φ_a (overdispersion)", f"{dc.phi_a:.3f}")
            col4.metric("ρ (bivariate corr)", f"{dc.rho:.4f}")

            # Parameter table
            if dc.alpha:
                rows = []
                for tid in sorted(dc.alpha.keys()):
                    rows.append({
                        "Team ID": tid,
                        "α (offense)": f"{dc.alpha[tid]:.4f}",
                        "β (defense)": f"{dc.beta.get(tid, 1.0):.4f}",
                        "Off Rank": "",
                        "Def Rank": "",
                    })
                df = pd.DataFrame(rows)
                alphas = [dc.alpha[t] for t in sorted(dc.alpha.keys())]
                betas = [dc.beta.get(t, 1.0) for t in sorted(dc.alpha.keys())]
                ranks_off = pd.Series(alphas).rank(ascending=False).astype(int).values
                ranks_def = pd.Series(betas).rank(ascending=True).astype(int).values
                df["Off Rank"] = ranks_off
                df["Def Rank"] = ranks_def

                st.subheader("Team Strength Parameters")
                st.dataframe(df, hide_index=True, use_container_width=True)

    # ── Tab 2: Temporal Decay ──────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("B.2 — Temporal Decay Weighting: w(Δt) = exp(−ξ × Δt)")
        st.latex(r"w(\Delta t) = e^{-\xi \cdot \Delta t}, \quad \xi = 0.0045 \text{ days}^{-1}")

        delta_range = np.arange(0, 400, 1)
        weights = np.array([temporal_weight(d) for d in delta_range])

        col_chart, col_info = st.columns([3, 1])
        with col_chart:
            df_decay = pd.DataFrame({"Days ago (Δt)": delta_range, "Weight w(Δt)": weights})
            st.line_chart(df_decay.set_index("Days ago (Δt)"))

        with col_info:
            st.markdown("**Key thresholds:**")
            for d in [0, 30, 60, 90, 180, 365]:
                w = temporal_weight(d)
                st.write(f"Δt={d:4d}d → w={w:.4f}")
            st.markdown(f"**Cutoff:** Δt > 1022d (w < 0.01)")
            st.markdown(f"**ξ = {XI}** (faster than football's 0.0065 due to NBA roster churn)")

    # ── Tab 3: Parameter Explorer ──────────────────────────────────────────────
    with tabs[2]:
        st.subheader("B.1 — Attack/Defense Strength Explorer")
        st.latex(r"\lambda_{i,j} = \alpha_i \times \beta_j \times \gamma^{H_{ij}} \times L_{avg}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Home Team**")
            alpha_home = st.slider("α_home (offense)", 0.5, 2.0, 1.10, 0.01)
            beta_home = st.slider("β_home (defense)", 0.5, 2.0, 0.95, 0.01)
        with c2:
            st.markdown("**Away Team**")
            alpha_away = st.slider("α_away (offense)", 0.5, 2.0, 0.92, 0.01)
            beta_away = st.slider("β_away (defense)", 0.5, 2.0, 1.05, 0.01)

        gamma = st.slider("γ (home court)", 0.8, 1.2, float(GAMMA_INIT), 0.005)
        l_avg_exp = st.slider("L_avg (pts/100 poss)", 100.0, 120.0, float(L_AVG_DEFAULT))
        pace = st.slider("Pace (poss/48 min)", 80.0, 115.0, 98.0)

        lambda_home = alpha_home * beta_away * gamma * l_avg_exp
        lambda_away = alpha_away * beta_home * 1.0 * l_avg_exp
        Lambda_home = lambda_home * (pace / 100.0)
        Lambda_away = lambda_away * (pace / 100.0)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("λ_home (pts/100 poss)", f"{lambda_home:.1f}")
        r2.metric("λ_away (pts/100 poss)", f"{lambda_away:.1f}")
        r3.metric("Λ_home (total expected pts)", f"{Lambda_home:.1f}")
        r4.metric("Λ_away (total expected pts)", f"{Lambda_away:.1f}")

        st.markdown("**Projected Total:** " + f"{Lambda_home + Lambda_away:.1f} pts  |  **Spread:** {Lambda_home - Lambda_away:+.1f}")

        # Bivariate correction demo
        st.markdown("---")
        st.subheader("B.3 — Bivariate Correction τ(S_h, S_a, ρ)")
        st.latex(r"\tau(S_h, S_a, \rho) = 1 + \rho \cdot f_{close}(S_h, S_a)")
        rho_demo = st.slider("ρ (bivariate correlation)", -0.15, 0.15, 0.0, 0.01)
        s_h = st.number_input("S_h (home final score)", 80, 150, 108)
        s_a = st.number_input("S_a (away final score)", 80, 150, 104)
        diff = abs(s_h - s_a)
        f_close = 1.0 if diff <= 6 else 0.0
        tau = 1.0 + rho_demo * f_close
        st.info(f"|S_h − S_a| = {diff} → f_close = {f_close:.0f} → τ = {tau:.4f}")
        if f_close == 1.0:
            st.caption("Close game detected: bivariate correction applied")
        else:
            st.caption("Blowout: bivariate correction inactive (f_close = 0)")

    # ── Tab 4: Game Records ────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("B.5 — Game Records & Incremental Newton Update")

        with st.expander("Add new game result (Newton step update)"):
            nc1, nc2 = st.columns(2)
            with nc1:
                home_id = st.number_input("Home Team ID", 1, 30, 1)
                away_id = st.number_input("Away Team ID", 1, 30, 2)
            with nc2:
                score_h = st.number_input("Home Score", 70, 160, 108)
                score_a = st.number_input("Away Score", 70, 160, 104)

            if st.button("Apply Newton Step Update"):
                if st.session_state.dc_parameters:
                    dc_state = st.session_state.dc_parameters
                    dc_obj = DCParameters(
                        alpha=dc_state.get("alpha", {home_id: 1.0, away_id: 1.0}),
                        beta=dc_state.get("beta", {home_id: 1.0, away_id: 1.0}),
                        gamma=dc_state.get("gamma", GAMMA_INIT),
                        phi_h=dc_state.get("phi_h", 2.0),
                        phi_a=dc_state.get("phi_a", 2.0),
                        rho=dc_state.get("rho", 0.0),
                        l_avg=dc_state.get("l_avg", L_AVG_DEFAULT),
                    )
                    new_game = GameRecord(
                        home_team_id=home_id, away_team_id=away_id,
                        score_home=score_h, score_away=score_a,
                        date=datetime.utcnow(),
                    )
                    updated_dc = incremental_newton_update(dc_obj, new_game)
                    st.session_state.dc_parameters.update({
                        "alpha": updated_dc.alpha,
                        "beta": updated_dc.beta,
                        "n_games": updated_dc.n_games,
                    })
                    st.success(f"✅ Newton step applied. New α_{home_id}={updated_dc.alpha.get(home_id, 1.0):.4f}, α_{away_id}={updated_dc.alpha.get(away_id, 1.0):.4f}")
                else:
                    st.warning("Please fit DC parameters first (Tab 1).")

        # Staleness guard status
        st.markdown("**B.5 — Staleness Guard**")
        last_data = st.session_state.get("last_data_ingested", datetime.utcnow() - timedelta(hours=12))
        hours_since = (datetime.utcnow() - last_data).total_seconds() / 3600 if isinstance(last_data, datetime) else 12.0
        if hours_since > 72:
            st.error(f"⚠️ STALENESS_FLAG ACTIVE — {hours_since:.0f}h since last data. Min EV threshold increased by +0.5pp.")
            st.session_state.staleness_flag = True
        else:
            st.success(f"✅ Data fresh — {hours_since:.1f}h since last ingestion (threshold: 72h)")
            st.session_state.staleness_flag = False
