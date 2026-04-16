"""NEXOM Dashboard — System overview and live status."""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def render():
    st.title("🏀 NEXOM — Basketball Betting Intelligence v5.0")
    st.caption("BLOOM × SILVER × BENTER × DIXON–COLES BASKETBALL HYBRID")

    # ── System status row ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Bankroll", f"${st.session_state.bankroll:,.2f}",
                  delta=f"+${np.random.uniform(10, 200):.2f}" if np.random.random() > 0.4 else f"-${np.random.uniform(10, 100):.2f}")
    with col2:
        st.metric("Regime", st.session_state.regime)
    with col3:
        st.metric("Kelly Multiplier", f"{st.session_state.kelly_fraction_multiplier:.0%}")
    with col4:
        active_stops = len([s for s in st.session_state.active_stops])
        st.metric("Active Stops", active_stops,
                  delta="⚠️ ALERT" if active_stops > 0 else "✅ Clear",
                  delta_color="inverse" if active_stops > 0 else "normal")
    with col5:
        drift = st.session_state.drift_status.get("active", False)
        st.metric("Drift Status", "🔴 ACTIVE" if drift else "✅ Nominal")

    st.markdown("---")

    # ── Performance metrics ────────────────────────────────────────────────────
    st.subheader("📊 Rolling Performance Metrics")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        brier = st.session_state.get("brier_30d", 0.218)
        st.metric("Brier Score 30d", f"{brier:.4f}",
                  delta=f"{'✅ ' if brier < 0.230 else '⚠️ '} target < 0.230",
                  delta_color="normal" if brier < 0.230 else "inverse")
    with m_col2:
        clv = st.session_state.get("clv_30d", 0.014)
        st.metric("CLV 30d", f"{clv:.3f}",
                  delta=f"{'✅ ' if clv > 0.012 else '⚠️ '} target > 0.012",
                  delta_color="normal" if clv > 0.012 else "inverse")
    with m_col3:
        roi = st.session_state.get("roi_30d", 0.024)
        st.metric("ROI 30d", f"{roi:.1%}",
                  delta=f"{'✅ ' if roi > 0.02 else '⚠️ '} target > 2%",
                  delta_color="normal" if roi > 0.02 else "inverse")
    with m_col4:
        ece = st.session_state.get("ece_30d", 0.032)
        st.metric("ECE 30d", f"{ece:.4f}",
                  delta=f"{'✅ ' if ece < 0.040 else '⚠️ '} target < 0.040",
                  delta_color="normal" if ece < 0.040 else "inverse")

    st.markdown("---")

    # ── P&L chart ──────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("💰 Cumulative P&L (90-Day)")
        dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
        np.random.seed(42)
        daily_pnl = np.random.normal(25, 120, 90)
        daily_pnl[15] = -380
        daily_pnl[41] = -220
        cumulative = np.cumsum(daily_pnl)
        df_pnl = pd.DataFrame({
            "Date": dates,
            "Cumulative P&L ($)": cumulative + st.session_state.bankroll - cumulative[-1],
        })
        st.line_chart(df_pnl.set_index("Date"))

    with col_right:
        st.subheader("📈 Market Coverage")
        df_markets = pd.DataFrame({
            "Market": ["Moneyline", "Spread", "Total", "Player Props"],
            "Bets": [145, 203, 187, 94],
            "Win%": ["54.5%", "52.2%", "53.7%", "51.1%"],
            "ROI": ["3.2%", "2.4%", "2.9%", "1.8%"],
        })
        st.dataframe(df_markets, hide_index=True, use_container_width=True)

    st.markdown("---")

    # ── Recent bets ────────────────────────────────────────────────────────────
    st.subheader("🎯 Recent Bet Signals")
    bets_data = []
    markets = ["moneyline", "spread", "total", "player_prop"]
    books = ["pinnacle", "circa", "bet365", "fanduel"]
    np.random.seed(99)
    for i in range(8):
        p_model = np.random.uniform(0.52, 0.65)
        p_market = p_model - np.random.uniform(0.03, 0.08)
        ev = np.random.uniform(0.025, 0.08)
        kelly = np.random.uniform(0.04, 0.15)
        bets_data.append({
            "Game": f"Team {chr(65+i)} vs Team {chr(73+i)}",
            "Market": np.random.choice(markets),
            "p_model": f"{p_model:.3f}",
            "p_market": f"{p_market:.3f}",
            "Edge": f"{(p_model-p_market):.3f}",
            "EV_exec": f"{ev:.3f}",
            "Kelly": f"{kelly:.3f}",
            "Book": np.random.choice(books),
            "Gates": "✅ All Pass" if np.random.random() > 0.3 else "❌ Failed",
        })

    df_bets = pd.DataFrame(bets_data)
    st.dataframe(
        df_bets.style.apply(
            lambda col: ["background-color: #1a3a1a" if v == "✅ All Pass" else "background-color: #3a1a1a" for v in col]
            if col.name == "Gates" else [""] * len(col),
            axis=0,
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("---")

    # ── System architecture overview ───────────────────────────────────────────
    st.subheader("🏗️ System Architecture — Service Status")
    services = {
        "Data Ingestion": {"status": "🟢 Running", "latency": "142ms", "budget": "200ms"},
        "Model Compute": {"status": "🟢 Running", "latency": "387ms", "budget": "800ms"},
        "Execution": {"status": "🟢 Running", "latency": "98ms", "budget": "150ms"},
        "Calibration & Learning": {"status": "🟢 Scheduled", "latency": "nightly", "budget": "30min post-game"},
        "Monitoring & Drift": {"status": "🟢 Running", "latency": "continuous", "budget": "Prometheus :9090"},
    }
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    for idx, (name, info) in enumerate(services.items()):
        col = [s_col1, s_col2, s_col3, s_col4][idx % 4]
        with col:
            st.markdown(f"**{name}**")
            st.write(f"Status: {info['status']}")
            st.write(f"Latency: `{info['latency']}` / `{info['budget']}`")

    st.markdown("---")
    st.caption("NEXOM v5.0 | Bloom × Silver × Benter × Dixon–Coles | Basketball Only")
