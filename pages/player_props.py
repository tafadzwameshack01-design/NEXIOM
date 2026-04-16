"""NEXOM — Player Props Model Page (Section D)."""
import streamlit as st
import numpy as np
import pandas as pd

from engines.player_props import (
    PlayerData, project_stat, prop_over_probability,
    gaussian_copula_simulate, estimate_correlation_matrix,
    generate_demo_player, PHI_BY_STAT, LEAGUE_AVG_RATES,
)


STATS = ["PTS", "REB", "AST", "STL", "BLK", "3PM"]


def render():
    st.title("👤 Player Prop Model")
    st.caption("Section D — Per-Player Per-Stat Projections, Bayesian Rates, Gaussian Copula")

    tabs = st.tabs(["📐 Projection Engine", "🔗 Copula Simulation", "📊 Prop O/U Calculator", "🎮 Multi-Player"])

    # ── D.1 Projection Engine ──────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("D.1 — Per-Player Per-Stat Projection")
        st.latex(r"\text{proj}(i,s) = \text{rate}(i,s) \times \text{min\_proj}(i) \times \text{matchup} \times \text{usage} \times \text{rest} \times \text{lineup}")

        col1, col2 = st.columns(2)
        with col1:
            player_name = st.text_input("Player name", "Demo Player")
            position = st.selectbox("Position", ["PG", "SG", "SF", "PF", "C"])
            season_mins = st.slider("Season avg minutes", 10.0, 42.0, 30.0, 0.5)
            is_b2b = st.checkbox("Back-to-back tonight")
            is_load = st.checkbox("Load management risk")
            days_rest = st.selectbox("Days rest", [0, 1, 2, 3, 4, 5], index=2)
            usage_pct = st.slider("Current USG%", 0.10, 0.40, 0.22, 0.01)
            baseline_usage = st.slider("Baseline USG%", 0.10, 0.40, 0.22, 0.01)

        with col2:
            stat_sel = st.selectbox("Stat to project", STATS)
            opp_def_rate = st.number_input(
                f"Opponent def rate vs position ({stat_sel})",
                0.0, 5.0,
                float(LEAGUE_AVG_RATES.get(stat_sel, 1.0) * 1.05),
                0.05,
            )
            prop_line = st.number_input(f"Prop line ({stat_sel})", 0.0, 60.0,
                                        {"PTS": 22.5, "REB": 7.5, "AST": 5.5, "STL": 1.5, "BLK": 1.5, "3PM": 2.5}.get(stat_sel, 10.5),
                                        0.5)

        # Generate demo player with realistic game log
        demo = generate_demo_player(player_id=1, name=player_name)
        demo.position = position
        demo.season_avg_minutes = season_mins
        demo.is_b2b = is_b2b
        demo.is_load_management_risk = is_load
        demo.days_rest = days_rest
        demo.usage_pct = usage_pct
        demo.baseline_usage_pct = baseline_usage

        if st.button("🎯 Run Projection", type="primary"):
            proj = project_stat(demo, stat_sel, opp_def_rate=opp_def_rate)
            p_over = prop_over_probability(proj.projection, prop_line, PHI_BY_STAT[stat_sel])

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric(f"Projection ({stat_sel})", f"{proj.projection:.2f}")
            pc2.metric("Prop line", f"{prop_line}")
            pc3.metric("P(Over)", f"{p_over:.4f}", f"{p_over*100:.1f}%")
            pc4.metric("Edge vs 50/50", f"{(p_over-0.5)*100:+.1f}pp")

            factor_df = pd.DataFrame([
                {"Factor": "Minutes projection (μ)", "Value": f"{proj.minutes_mean:.2f} min"},
                {"Factor": "Matchup factor", "Value": f"{proj.matchup_factor:.4f}"},
                {"Factor": "Usage factor", "Value": f"{proj.usage_factor:.4f}"},
                {"Factor": "Rest factor", "Value": f"{proj.rest_factor:.4f}"},
                {"Factor": "Lineup factor", "Value": f"{proj.lineup_factor:.4f}"},
            ])
            st.dataframe(factor_df, hide_index=True, use_container_width=True)

            # All-stat projections
            st.markdown("---")
            st.subheader("Full Stat Projection Profile")
            all_projs = []
            for s in STATS:
                p = project_stat(demo, s, opp_def_rate=LEAGUE_AVG_RATES.get(s, 1.0) * 1.02)
                all_projs.append({"Stat": s, "Projection": f"{p.projection:.2f}",
                                   "φ (overdispersion)": PHI_BY_STAT[s]})
            st.dataframe(pd.DataFrame(all_projs), hide_index=True, use_container_width=True)

    # ── D.2 Copula Simulation ──────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("D.2 — Gaussian Copula Joint Distribution")
        st.latex(r"Z_i \sim \mathcal{N}(0, \Sigma_i), \quad U_i = \Phi(Z_i), \quad X_i = F_i^{-1}(U_i)")

        col1, col2 = st.columns(2)
        with col1:
            n_cop_samples = st.select_slider("N simulation samples", [1000, 5000, 10000], value=5000)
            st.markdown("**Base projections per stat:**")
            projections = {}
            for s in STATS:
                default = {"PTS": 22.0, "REB": 7.0, "AST": 5.0, "STL": 1.2, "BLK": 0.8, "3PM": 2.5}.get(s, 5.0)
                projections[s] = st.number_input(f"{s} projection", 0.0, 60.0, default, 0.5, key=f"cop_{s}")

        if st.button("🔗 Run Copula Simulation", type="primary"):
            demo_cop = generate_demo_player(player_id=99)
            sigma = estimate_correlation_matrix(demo_cop.game_log)
            samples = gaussian_copula_simulate(projections, sigma, n_cop_samples)

            with col2:
                st.markdown("**Correlation Matrix (Ledoit-Wolf)**")
                corr_df = pd.DataFrame(sigma, index=STATS, columns=STATS)
                st.dataframe(corr_df.applymap(lambda x: f"{x:.3f}").style.background_gradient(cmap="RdYlGn"), use_container_width=True)

            # Summary statistics
            st.markdown("---")
            st.subheader("Simulation Summary")
            summary_rows = []
            for j, s in enumerate(STATS):
                col_data = samples[:, j]
                summary_rows.append({
                    "Stat": s,
                    "Mean": f"{col_data.mean():.2f}",
                    "Std": f"{col_data.std():.2f}",
                    "5th": f"{np.percentile(col_data, 5):.1f}",
                    "25th": f"{np.percentile(col_data, 25):.1f}",
                    "Median": f"{np.median(col_data):.1f}",
                    "75th": f"{np.percentile(col_data, 75):.1f}",
                    "95th": f"{np.percentile(col_data, 95):.1f}",
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

            st.session_state["copula_samples"] = samples

    # ── Prop O/U Calculator ────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Prop Over/Under Probability Calculator")

        pu_col1, pu_col2 = st.columns(2)
        with pu_col1:
            stat_pu = st.selectbox("Stat", STATS, key="pu_stat")
            proj_pu = st.number_input("Projection", 0.1, 60.0, 22.5, 0.5, key="pu_proj")
            line_pu = st.number_input("Prop line", 0.5, 60.0, 21.5, 0.5, key="pu_line")
            phi_pu = st.number_input("φ (overdispersion)", 0.5, 5.0, float(PHI_BY_STAT.get(stat_pu, 2.0)), 0.1, key="pu_phi")
            n_pu = st.select_slider("N samples", [5000, 10000, 50000], value=10000, key="pu_n")

        if st.button("Calculate P(Over) & P(Under)", type="primary", key="pu_btn"):
            p_over = prop_over_probability(proj_pu, line_pu, phi_pu, n_pu)
            p_under = 1.0 - p_over

            with pu_col2:
                st.metric("P(Over)", f"{p_over:.4f}", f"{p_over*100:.2f}%")
                st.metric("P(Under)", f"{p_under:.4f}", f"{p_under*100:.2f}%")
                st.metric("Projection vs line", f"{proj_pu - line_pu:+.1f}")

                # Fair decimal odds
                fair_over = 1.0 / p_over if p_over > 0 else 999
                fair_under = 1.0 / p_under if p_under > 0 else 999
                st.metric("Fair odds (Over)", f"{fair_over:.3f}")
                st.metric("Fair odds (Under)", f"{fair_under:.3f}")

    # ── Multi-Player ───────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Multi-Player Projection Dashboard")

        n_players = st.slider("Number of players", 2, 8, 4)
        player_rows = []
        for i in range(n_players):
            np.random.seed(i * 7)
            demo_p = generate_demo_player(player_id=i+1, name=f"Player {chr(65+i)}")
            all_stat_projs = {}
            for s in STATS:
                p = project_stat(demo_p, s)
                all_stat_projs[s] = p.projection
            row = {"Player": f"Player {chr(65+i)}"}
            row.update({s: f"{all_stat_projs[s]:.1f}" for s in STATS})
            player_rows.append(row)

        df_multi = pd.DataFrame(player_rows)
        st.dataframe(df_multi, hide_index=True, use_container_width=True)

        st.caption("Projections based on Bayesian posterior mean rates × minutes projection × all adjustment factors (D.1)")
