import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NEXOM — Basketball Betting Intelligence v5.0",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 10000.0
if "kelly_fraction_multiplier" not in st.session_state:
    st.session_state.kelly_fraction_multiplier = 0.30
if "regime" not in st.session_state:
    st.session_state.regime = "REGIME_NORMAL"
if "active_stops" not in st.session_state:
    st.session_state.active_stops = []
if "dc_parameters" not in st.session_state:
    st.session_state.dc_parameters = {}
if "placed_bets" not in st.session_state:
    st.session_state.placed_bets = []
if "resolved_bets" not in st.session_state:
    st.session_state.resolved_bets = []
if "rejected_bets" not in st.session_state:
    st.session_state.rejected_bets = []
if "calibration_weights" not in st.session_state:
    st.session_state.calibration_weights = {"alpha": 0.45, "beta": 0.40, "gamma": 0.15}
if "bookmaker_efficiency" not in st.session_state:
    st.session_state.bookmaker_efficiency = {}
if "steam_events" not in st.session_state:
    st.session_state.steam_events = []
if "risk_events" not in st.session_state:
    st.session_state.risk_events = []
if "simulation_cache" not in st.session_state:
    st.session_state.simulation_cache = {}
if "drift_status" not in st.session_state:
    st.session_state.drift_status = {"active": False, "metrics": {}}
if "regime_history" not in st.session_state:
    st.session_state.regime_history = []
if "stop_loop" not in st.session_state:
    st.session_state.stop_loop = False
if "improvement_iteration" not in st.session_state:
    st.session_state.improvement_iteration = 0
if "staleness_flag" not in st.session_state:
    st.session_state.staleness_flag = False
if "calibration_failure_flags" not in st.session_state:
    st.session_state.calibration_failure_flags = {}
if "live_game_state" not in st.session_state:
    st.session_state.live_game_state = None
if "simulation_outputs" not in st.session_state:
    st.session_state.simulation_outputs = {}
if "portfolio_bets" not in st.session_state:
    st.session_state.portfolio_bets = []
if "pnl_history" not in st.session_state:
    st.session_state.pnl_history = []

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/300x60/0d1117/00ff88?text=NEXOM+v5.0", use_container_width=True)
st.sidebar.markdown("---")

pages = {
    "🏀 Dashboard": "dashboard",
    "📊 Dixon–Coles Model": "dc_model",
    "🎯 Monte Carlo Engine": "mc_engine",
    "📈 Market Probability": "market_engine",
    "⚡ Execution Microstructure": "execution",
    "🔧 Calibration Engine": "calibration",
    "💼 Portfolio Optimisation": "portfolio",
    "🛡️ Risk Management": "risk",
    "🔄 Self-Improving Loop": "self_improving",
    "✅ Bet Validity Gate": "validity_gate",
    "👤 Player Props": "player_props",
    "⚙️ System Config": "config",
}

if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"

for label, key in pages.items():
    if st.sidebar.button(label, key=f"nav_{key}", use_container_width=True):
        st.session_state.current_page = key

st.sidebar.markdown("---")
st.sidebar.metric("Bankroll", f"${st.session_state.bankroll:,.2f}")
st.sidebar.metric("Regime", st.session_state.regime)
st.sidebar.metric("Kelly Multiplier", f"{st.session_state.kelly_fraction_multiplier:.2f}")

if st.session_state.active_stops:
    st.sidebar.error(f"🔴 ACTIVE STOPS: {len(st.session_state.active_stops)}")

# ── Page routing ──────────────────────────────────────────────────────────────
page = st.session_state.current_page

if page == "dashboard":
    from pages import dashboard
    dashboard.render()
elif page == "dc_model":
    from pages import dc_model
    dc_model.render()
elif page == "mc_engine":
    from pages import mc_engine
    mc_engine.render()
elif page == "market_engine":
    from pages import market_engine
    market_engine.render()
elif page == "execution":
    from pages import execution
    execution.render()
elif page == "calibration":
    from pages import calibration
    calibration.render()
elif page == "portfolio":
    from pages import portfolio
    portfolio.render()
elif page == "risk":
    from pages import risk
    risk.render()
elif page == "self_improving":
    from pages import self_improving
    self_improving.render()
elif page == "validity_gate":
    from pages import validity_gate
    validity_gate.render()
elif page == "player_props":
    from pages import player_props
    player_props.render()
elif page == "config":
    from pages import config
    config.render()
