"""NEXOM — Risk Management System Page (Section J)."""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from engines.risk import (
    RiskState, run_all_risk_checks, compute_ruin_probability,
    ActiveStop, check_7d_drawdown, check_daily_loss,
    check_category_streak, check_volatility_spike,
)


def render():
    st.title("🛡️ Risk Management System")
    st.caption("Section J — Hard Stops, Ruin Probability Guard, Drawdown Tracking")

    tabs = st.tabs(["🚨 Stop Conditions", "☠️ Ruin Guard", "📉 Drawdown Tracker", "⚙️ Risk Simulator"])

    # ── J.1 Hard Stops ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("J.1 — Hard Stop Conditions")
        st.info("All conditions evaluated continuously after every resolved bet. Hard stops do NOT cancel submitted bets.")

        col1, col2 = st.columns(2)
        with col1:
            bankroll = st.number_input("Current bankroll ($)", 100.0, 1000000.0, st.session_state.bankroll, 100.0)
            bankroll_7d = st.number_input("Bankroll 7 days ago ($)", 100.0, 1000000.0, bankroll * 1.05, 100.0)
            bankroll_30d = st.number_input("Bankroll 30 days ago ($)", 100.0, 1000000.0, bankroll * 1.12, 100.0)
            daily_pnl = st.number_input("Today's P&L ($)", -10000.0, 10000.0, -150.0, 10.0)

        with col2:
            n_daily_pnl = st.slider("Days of P&L history", 7, 90, 30)
            np.random.seed(5)
            pnl_history = list(np.random.normal(20, 150, n_daily_pnl))

            category_sel = st.selectbox("Market category to check", ["moneyline", "spread", "total", "player_prop"])
            consecutive_losses = st.slider("Consecutive losses in category", 0, 10, 4)

        state = RiskState(
            bankroll=bankroll,
            bankroll_7d_ago=bankroll_7d,
            bankroll_30d_ago=bankroll_30d,
            daily_pnl=daily_pnl,
            active_stops=list(st.session_state.active_stops),
            kelly_fraction_multiplier=st.session_state.kelly_fraction_multiplier,
            pnl_history_90d=pnl_history,
            win_count_30d=20,
            loss_count_30d=15,
            category_streaks={category_sel: consecutive_losses},
        )

        if st.button("🔍 Run All Risk Checks", type="primary"):
            updated_state, triggered = run_all_risk_checks(state, category=category_sel, avg_stake=200.0)
            st.session_state.active_stops = updated_state.active_stops
            st.session_state.kelly_fraction_multiplier = updated_state.kelly_fraction_multiplier

            if triggered:
                for t in triggered:
                    st.error(f"🔴 TRIGGERED: {t}")
            else:
                st.success("✅ No new stops triggered. System operating normally.")

        # Stop status display
        st.markdown("---")
        st.subheader("Active Stops")
        active = [s for s in st.session_state.active_stops if s.is_active]
        if active:
            for stop in active:
                with st.expander(f"🔴 {stop.stop_type} — {stop.remaining_hours:.1f}h remaining"):
                    st.json({
                        "type": stop.stop_type,
                        "triggered_at": stop.triggered_at.isoformat(),
                        "expires_at": stop.expires_at.isoformat(),
                        "remaining_hours": f"{stop.remaining_hours:.1f}h",
                        "details": stop.details,
                    })
        else:
            st.success("✅ No active stops")

        # Stop reference table
        st.markdown("---")
        stops_ref = pd.DataFrame([
            {"Stop": "STOP_7D_DRAWDOWN", "Trigger": "7d drawdown > 12% of bankroll-at-risk", "Duration": "48 hours", "Scope": "All betting"},
            {"Stop": "STOP_DAILY_LOSS", "Trigger": "Daily P&L < −4% of bankroll", "Duration": "Until UTC midnight", "Scope": "All betting"},
            {"Stop": "STOP_CATEGORY_STREAK", "Trigger": "7 consecutive losses in one category", "Duration": "48 hours", "Scope": "That category only"},
            {"Stop": "STOP_VOLATILITY_SPIKE", "Trigger": "7d P&L std > 3× 90d baseline std", "Duration": "72 hours + Kelly ×0.5", "Scope": "All betting"},
        ])
        st.dataframe(stops_ref, hide_index=True, use_container_width=True)

    # ── J.2 Ruin Guard ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("J.2 — Ruin Probability Guard")
        st.latex(r"P(\text{ruin}) \approx \left(\frac{1 - p_{edge}}{p_{edge}}\right)^{\text{bankroll}/\text{avg\_stake}}")

        col1, col2 = st.columns(2)
        with col1:
            bankroll_r = st.number_input("Bankroll ($)", 100.0, 1000000.0, float(st.session_state.bankroll), 100.0, key="ruin_br")
            avg_stake_r = st.number_input("Average stake ($)", 1.0, 10000.0, 200.0, 10.0, key="ruin_stake")
            win_rate_r = st.slider("Win rate 30d", 0.30, 0.70, 0.53, 0.01, key="ruin_wr")

        p_ruin = compute_ruin_probability(bankroll_r, avg_stake_r, win_rate_r)

        with col2:
            color = "inverse" if p_ruin > 0.002 else "normal"
            st.metric("P(ruin)", f"{p_ruin:.6f}", f"{p_ruin*100:.4f}%", delta_color=color)
            st.metric("Threshold", "0.002 (0.2%)")

            if p_ruin > 0.002:
                reduction = max(0.0, st.session_state.kelly_fraction_multiplier - 0.05)
                st.error(f"⚠️ RUIN_GUARD ACTIVE: Kelly multiplier reduced from {st.session_state.kelly_fraction_multiplier:.2f} → {reduction:.2f}")
            else:
                st.success("✅ Ruin probability within safe bounds")

        # Sensitivity surface
        st.markdown("---")
        st.subheader("Ruin Probability Sensitivity")
        stake_range = np.arange(50, 500, 50)
        win_range = np.arange(0.48, 0.60, 0.02)
        ruin_grid = np.array([[compute_ruin_probability(bankroll_r, s, w) for w in win_range] for s in stake_range])
        df_ruin = pd.DataFrame(
            ruin_grid,
            index=[f"stake=${s}" for s in stake_range],
            columns=[f"win={w:.2f}" for w in win_range],
        )
        st.dataframe(df_ruin.applymap(lambda x: f"{x:.6f}").style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)

    # ── Drawdown Tracker ───────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Drawdown Tracker — 90-Day Rolling")

        np.random.seed(12)
        n_days = 90
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")
        daily_returns = np.random.normal(0.002, 0.025, n_days)
        bankroll_curve = st.session_state.bankroll * np.cumprod(1 + daily_returns)
        rolling_max = np.maximum.accumulate(bankroll_curve)
        drawdown = (rolling_max - bankroll_curve) / rolling_max

        df_dd = pd.DataFrame({
            "Date": dates,
            "Bankroll ($)": bankroll_curve,
            "Drawdown (%)": drawdown * 100,
        })

        col_l, col_r = st.columns(2)
        with col_l:
            st.line_chart(df_dd.set_index("Date")["Bankroll ($)"])
        with col_r:
            st.area_chart(df_dd.set_index("Date")["Drawdown (%)"])

        max_dd = float(drawdown.max())
        dd_7d = float(drawdown[-7:].max())
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Max Drawdown 90d", f"{max_dd*100:.2f}%")
        dc2.metric("Current 7d Drawdown", f"{dd_7d*100:.2f}%",
                   delta="⚠️ STOP triggered" if dd_7d > 0.12 else "✅ Within limit",
                   delta_color="inverse" if dd_7d > 0.12 else "normal")
        dc3.metric("STOP threshold", "12%")

    # ── Risk Simulator ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Risk Scenario Simulator")
        st.caption("Test which stops would trigger under a given scenario")

        sc1, sc2 = st.columns(2)
        with sc1:
            sim_bankroll = st.number_input("Bankroll ($)", 1000.0, 1000000.0, 10000.0, 500.0, key="sim_br")
            sim_bankroll_7d = st.number_input("Bankroll 7d ago ($)", 1000.0, 1000000.0, 11500.0, 500.0, key="sim_7d")
            sim_daily_pnl = st.number_input("Today's P&L ($)", -5000.0, 5000.0, -500.0, 50.0, key="sim_pnl")
        with sc2:
            sim_streak = st.slider("Consecutive losses in category", 0, 10, 8, key="sim_streak")
            sim_vol = st.slider("7d P&L std vs 90d std ratio", 0.5, 5.0, 3.5, 0.1)
            sim_win_rate = st.slider("30d win rate", 0.30, 0.70, 0.48, 0.01, key="sim_wr")
            sim_avg_stake = st.number_input("Avg stake ($)", 10.0, 5000.0, 300.0, 10.0, key="sim_stake")

        if st.button("🎭 Simulate Risk Scenario"):
            results = []

            # 7d drawdown
            dd = (sim_bankroll_7d - sim_bankroll) / max(sim_bankroll_7d, 1)
            results.append({"Stop": "STOP_7D_DRAWDOWN", "Value": f"{dd*100:.2f}%", "Threshold": "12%",
                            "Triggered": "🔴 YES" if dd > 0.12 else "✅ NO"})

            # Daily loss
            dl = -sim_daily_pnl / max(sim_bankroll, 1)
            results.append({"Stop": "STOP_DAILY_LOSS", "Value": f"{dl*100:.2f}%", "Threshold": "4%",
                            "Triggered": "🔴 YES" if dl > 0.04 else "✅ NO"})

            # Category streak
            results.append({"Stop": "STOP_CATEGORY_STREAK", "Value": f"{sim_streak} losses", "Threshold": "7",
                            "Triggered": "🔴 YES" if sim_streak >= 7 else "✅ NO"})

            # Volatility spike
            results.append({"Stop": "STOP_VOLATILITY_SPIKE", "Value": f"ratio={sim_vol:.1f}", "Threshold": "3×",
                            "Triggered": "🔴 YES" if sim_vol > 3.0 else "✅ NO"})

            # Ruin guard
            p_ruin_sim = compute_ruin_probability(sim_bankroll, sim_avg_stake, sim_win_rate)
            results.append({"Stop": "RUIN_GUARD", "Value": f"P(ruin)={p_ruin_sim:.5f}", "Threshold": "0.002",
                            "Triggered": "🔴 YES" if p_ruin_sim > 0.002 else "✅ NO"})

            df_res = pd.DataFrame(results)
            st.dataframe(df_res, hide_index=True, use_container_width=True)

            n_triggered = sum(1 for r in results if "YES" in r["Triggered"])
            if n_triggered > 0:
                st.error(f"🔴 {n_triggered} stop(s) would be triggered in this scenario")
            else:
                st.success("✅ No stops triggered — scenario within risk parameters")
