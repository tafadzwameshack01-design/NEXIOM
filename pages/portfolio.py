"""NEXOM — Portfolio Optimisation Engine Page (Section I)."""
import streamlit as st
import numpy as np
import pandas as pd

from engines.portfolio import (
    BetCandidate, PortfolioResult, optimise_portfolio,
    determine_regime, KELLY_MULTIPLIER_BY_REGIME,
)
from engines.execution import kelly_fraction


def render():
    st.title("💼 Portfolio Optimisation Engine (Benter Layer)")
    st.caption("Section I — Correlated Kelly, SLSQP Constraints, Regime Scaling")

    tabs = st.tabs(["📐 Kelly Calculator", "🔗 Correlated Portfolio", "🌡️ Regime Manager", "📊 Constraint Monitor"])

    # ── I.1 Kelly Calculator ───────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("I.1 — Kelly Criterion")
        st.latex(r"f^* = \frac{p \cdot d - 1}{d - 1}")
        st.latex(r"\text{stake} = f^* \times \text{kelly\_multiplier} \times \text{bankroll}")

        col1, col2 = st.columns(2)
        with col1:
            p_k = st.slider("p_final (calibrated)", 0.40, 0.80, 0.56, 0.01, key="k_p")
            d_k = st.number_input("Decimal odds", 1.5, 5.0, 1.92, 0.01, key="k_d")
            bankroll_k = st.number_input("Bankroll ($)", 1000.0, 1000000.0, float(st.session_state.bankroll), 100.0, key="k_br")

        with col2:
            kelly_mult = st.slider("Kelly multiplier", 0.10, 0.35, float(st.session_state.kelly_fraction_multiplier), 0.05)
            f_star = kelly_fraction(p_k, d_k, cap=0.25)
            stake = f_star * kelly_mult * bankroll_k
            ev_bet = p_k * d_k - 1.0

            st.metric("f* (full Kelly)", f"{f_star:.4f}", f"{f_star*100:.2f}%")
            st.metric("Fractional Kelly stake", f"${stake:.2f}")
            st.metric("EV per unit staked", f"{ev_bet:.4f}")
            if f_star <= 0:
                st.error("Negative Kelly: bet has negative EV — do not place")
            elif f_star > 0.25:
                st.warning(f"f*={f_star:.3f} capped at 0.25 (overbetting guard)")
            else:
                st.success(f"Valid Kelly stake: ${stake:.2f}")

        # Kelly fraction vs odds/probability surface
        st.markdown("---")
        st.subheader("Kelly Fraction Surface")
        p_range = np.arange(0.45, 0.75, 0.05)
        d_range = np.arange(1.5, 2.5, 0.1)
        kelly_grid = np.array([[kelly_fraction(p, d) for d in d_range] for p in p_range])
        df_kelly = pd.DataFrame(kelly_grid, index=[f"p={p:.2f}" for p in p_range],
                                columns=[f"d={d:.1f}" for d in d_range])
        st.dataframe(df_kelly.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

    # ── I.2 Correlated Portfolio ───────────────────────────────────────────────
    with tabs[1]:
        st.subheader("I.2 — Correlated Portfolio Kelly (SLSQP)")
        st.latex(r"\max \sum_i f_i \cdot EV_i - \frac{1}{2}\sum_{i,j} f_i f_j C_{ij}")
        st.caption("Subject to: total ≤ 35%, per-bet ≤ 8%, per-game ≤ 12%, per-book ≤ 20%, variance cap")

        n_bets = st.slider("Number of simultaneous bets", 2, 8, 4)

        bet_rows = []
        bets = []
        np.random.seed(42)

        for i in range(n_bets):
            cols = st.columns([2, 1, 1, 1, 1, 1])
            game = cols[0].selectbox(f"Game {i+1}", [f"Game {chr(65+j)}" for j in range(5)], index=i % 5, key=f"pg{i}")
            p_b = cols[1].number_input("p_final", 0.4, 0.8, round(0.50 + np.random.uniform(0.02, 0.12), 2), 0.01, key=f"pp{i}")
            d_b = cols[2].number_input("odds", 1.5, 3.0, round(1.85 + np.random.uniform(0, 0.15), 2), 0.01, key=f"pd{i}")
            book_b = cols[3].selectbox("book", ["pinnacle", "bet365", "fanduel"], index=i % 3, key=f"pb{i}")
            ev_b = cols[4].number_input("EV_exec", 0.01, 0.15, round(np.random.uniform(0.03, 0.08), 3), 0.005, key=f"pev{i}")
            f_b = kelly_fraction(p_b, d_b)
            stake_raw = f_b * st.session_state.kelly_fraction_multiplier * st.session_state.bankroll

            bets.append(BetCandidate(
                bet_id=f"bet_{i}",
                ev=ev_b, p_final=p_b, decimal_odds=d_b,
                kelly_raw=f_b, game_id=game.replace(" ", "_"),
                bookmaker=book_b,
                stake_raw=max(stake_raw, 1.0),
                outcome_sim=np.random.binomial(1, p_b, 1000).astype(float),
            ))
            bet_rows.append({"Bet": f"bet_{i}", "Game": game, "p": p_b, "odds": d_b,
                             "EV": ev_b, "f*": f"{f_b:.4f}", "Book": book_b, "Stake_raw": f"${stake_raw:.2f}"})

        if st.button("🔄 Optimise Portfolio (SLSQP)", type="primary"):
            regime = st.session_state.regime
            with st.spinner("Running SLSQP optimisation..."):
                result = optimise_portfolio(bets, st.session_state.bankroll, regime)

            status = "✅ SLSQP converged" if result.slsqp_converged else "⚠️ Fallback (proportional Kelly)"
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Total exposure", f"${result.total_exposure:.2f}",
                       f"{result.total_exposure/st.session_state.bankroll*100:.1f}% of bankroll")
            rc2.metric("Portfolio EV", f"{result.expected_portfolio_ev:.4f}")
            rc3.metric("Portfolio variance", f"{result.portfolio_variance:.6f}")
            rc4.metric("Solver", status)

            stake_data = []
            for bet in bets:
                opt_stake = result.stakes.get(bet.bet_id, 0.0)
                stake_data.append({
                    "Bet ID": bet.bet_id, "Game": bet.game_id,
                    "Raw stake": f"${bet.stake_raw:.2f}",
                    "Optimised stake": f"${opt_stake:.2f}",
                    "% bankroll": f"{result.fractions.get(bet.bet_id, 0)*100:.2f}%",
                })
            st.dataframe(pd.DataFrame(stake_data), hide_index=True, use_container_width=True)

    # ── I.3 Regime Manager ────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("I.3 — Regime-Based Kelly Scaling")

        clv_30d_in = st.slider("CLV 30d", -0.02, 0.04, 0.014, 0.001, format="%.3f")
        roi_30d_in = st.slider("ROI 30d", -0.08, 0.12, 0.025, 0.005, format="%.3f")

        regime = determine_regime(clv_30d_in, roi_30d_in)
        kelly_mult_regime = KELLY_MULTIPLIER_BY_REGIME[regime]

        regime_colors = {
            "REGIME_BULL": "🟢",
            "REGIME_NORMAL": "🔵",
            "REGIME_CAUTIOUS": "🟡",
            "REGIME_HALT": "🔴",
        }

        st.metric(f"{regime_colors[regime]} Current Regime", regime)
        st.metric("Kelly Multiplier", f"{kelly_mult_regime:.0%}")

        if regime == "REGIME_HALT":
            st.error("🔴 REGIME_HALT: Kelly multiplier = 0.0. No betting. Drift investigation triggered.")
        elif regime == "REGIME_CAUTIOUS":
            st.warning("🟡 REGIME_CAUTIOUS: Kelly reduced to 20%. CLV or ROI below threshold.")
        elif regime == "REGIME_BULL":
            st.success("🟢 REGIME_BULL: Kelly at maximum 35%. Performance above targets.")
        else:
            st.info("🔵 REGIME_NORMAL: Standard 30% Kelly.")

        # Update session state
        if st.button("Apply regime to session"):
            st.session_state.regime = regime
            st.session_state.kelly_fraction_multiplier = kelly_mult_regime
            st.success(f"Regime updated to {regime}")

        # Regime thresholds table
        st.markdown("**Regime Thresholds:**")
        regime_df = pd.DataFrame([
            {"Regime": "REGIME_BULL", "CLV 30d": "> 0.015", "ROI 30d": "> 0.06", "Kelly": "35%"},
            {"Regime": "REGIME_NORMAL", "CLV 30d": "> 0.010", "ROI 30d": "> 0.02", "Kelly": "30%"},
            {"Regime": "REGIME_CAUTIOUS", "CLV 30d": "< 0.010 or ROI < 0", "ROI 30d": "< 0.00", "Kelly": "20%"},
            {"Regime": "REGIME_HALT", "CLV 30d": "< 0.005", "ROI 30d": "< −0.04", "Kelly": "0% (HALT)"},
        ])
        st.dataframe(regime_df, hide_index=True, use_container_width=True)

    # ── Constraint Monitor ─────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("I.2 — SLSQP Constraint Monitor")
        bankroll = st.session_state.bankroll

        constraints = {
            "Total portfolio exposure (≤ 35%)": (0.35 * bankroll, "sum of all stakes"),
            "Per-bet exposure (≤ 8%)": (0.08 * bankroll, "per individual bet"),
            "Per-game exposure (≤ 12%)": (0.12 * bankroll, "sum of bets on same game"),
            "Per-bookmaker exposure (≤ 20%)": (0.20 * bankroll, "sum of bets at same book"),
            "Variance cap σ²_max": ((0.05 * bankroll) ** 2, "(5% × bankroll)²"),
        }

        rows = []
        for name, (limit, note) in constraints.items():
            rows.append({"Constraint": name, "Limit": f"${limit:,.2f}", "Note": note})

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Based on current bankroll: ${bankroll:,.2f}")
