"""NEXOM — System Configuration Page."""
import streamlit as st
import os
import pandas as pd
import numpy as np


def render():
    st.title("⚙️ System Configuration")
    st.caption("API keys, bankroll setup, system parameters, and operational settings")

    tabs = st.tabs(["🔑 API & Keys", "💰 Bankroll", "📐 Model Parameters", "🏗️ Architecture", "📖 Pydantic Schemas"])

    # ── API Keys ────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("API Configuration")

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            st.success(f"✅ ANTHROPIC_API_KEY loaded ({len(api_key)} chars)")
        else:
            st.error("❌ ANTHROPIC_API_KEY not found in environment")
            st.code("export ANTHROPIC_API_KEY=your_key_here", language="bash")

        st.markdown("**Required External API Keys (set in .env):**")
        api_table = pd.DataFrame([
            {"Key": "ANTHROPIC_API_KEY", "Used for": "Self-correcting loop Claude analysis", "Required": "✅ Yes"},
            {"Key": "ODDS_API_KEY", "Used for": "Odds API (the-odds-api.com v4) — 12 books, 15s refresh", "Required": "✅ Yes"},
            {"Key": "SPORTRADAR_API_KEY", "Used for": "Live play-by-play WebSocket stream", "Required": "✅ Yes"},
            {"Key": "PINNACLE_API_KEY", "Used for": "Sharp book anchor, closing lines", "Required": "✅ Yes"},
            {"Key": "ROTOWIRE_API_KEY", "Used for": "Injury feed, minutes projections, late-scratch alerts", "Required": "✅ Yes"},
            {"Key": "DATABASE_URL", "Used for": "PostgreSQL 15 connection string", "Required": "✅ Yes"},
            {"Key": "REDIS_URL", "Used for": "Redis 7.2 (Streams + cache)", "Required": "✅ Yes"},
            {"Key": "SLACK_WEBHOOK_URL", "Used for": "Drift/risk alerts", "Required": "🟡 Recommended"},
            {"Key": "S3_BUCKET_URL", "Used for": "Model checkpoint storage (Backblaze B2 / AWS)", "Required": "🟡 Recommended"},
        ])
        st.dataframe(api_table, hide_index=True, use_container_width=True)

    # ── Bankroll ────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Bankroll & Kelly Configuration")

        new_bankroll = st.number_input("Current bankroll ($)", 100.0, 10000000.0, float(st.session_state.bankroll), 100.0)
        if st.button("Update bankroll"):
            st.session_state.bankroll = new_bankroll
            st.success(f"Bankroll updated to ${new_bankroll:,.2f}")

        st.markdown("---")
        new_kelly = st.slider("Kelly fraction multiplier", 0.0, 0.35, float(st.session_state.kelly_fraction_multiplier), 0.05)
        if st.button("Update Kelly multiplier"):
            st.session_state.kelly_fraction_multiplier = new_kelly
            st.success(f"Kelly multiplier updated to {new_kelly:.2f}")

        st.markdown("**Exposure caps (I.2):**")
        caps_df = pd.DataFrame([
            {"Constraint": "Total portfolio", "Cap": "35% bankroll", "Value": f"${new_bankroll * 0.35:,.2f}"},
            {"Constraint": "Per-bet", "Cap": "8% bankroll", "Value": f"${new_bankroll * 0.08:,.2f}"},
            {"Constraint": "Per-game", "Cap": "12% bankroll", "Value": f"${new_bankroll * 0.12:,.2f}"},
            {"Constraint": "Per-bookmaker", "Cap": "20% bankroll", "Value": f"${new_bankroll * 0.20:,.2f}"},
            {"Constraint": "Max Kelly fraction", "Cap": "25% (f* cap)", "Value": "—"},
        ])
        st.dataframe(caps_df, hide_index=True, use_container_width=True)

    # ── Model Parameters ────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Core Model Parameters")

        param_df = pd.DataFrame([
            {"Parameter": "ξ (temporal decay)", "Value": "0.0045", "Units": "days⁻¹", "Source": "B.2 — faster than football's 0.0065"},
            {"Parameter": "γ (home court)", "Value": "1.035", "Units": "multiplicative", "Source": "B.1 — 3-4 pts/100 poss advantage"},
            {"Parameter": "φ_h initial", "Value": "2.0", "Units": "overdispersion", "Source": "B.3 — NegBin parameter"},
            {"Parameter": "ρ initial", "Value": "0.0", "Units": "corr ∈ [−0.15, 0.15]", "Source": "B.3 — bivariate correction"},
            {"Parameter": "N_pre", "Value": "100,000", "Units": "simulations", "Source": "E.1 — pre-game"},
            {"Parameter": "N_live", "Value": "25,000", "Units": "simulations", "Source": "E.1 — live"},
            {"Parameter": "σ_pace", "Value": "3.5", "Units": "poss/48min", "Source": "E.2 — pace uncertainty"},
            {"Parameter": "min_edge", "Value": "0.030", "Units": "probability", "Source": "G.1 — 3pp minimum edge"},
            {"Parameter": "kelly_multiplier", "Value": "0.30 (default)", "Units": "fraction", "Source": "I.1 — 30% fractional Kelly"},
            {"Parameter": "max_kelly_cap", "Value": "0.25", "Units": "fraction", "Source": "I.1 — overbetting guard"},
            {"Parameter": "α blend weight", "Value": "≈0.45", "Units": "weight", "Source": "H.1 — model weight"},
            {"Parameter": "β blend weight", "Value": "≈0.40", "Units": "weight", "Source": "H.1 — market weight"},
            {"Parameter": "γ blend weight", "Value": "≈0.15", "Units": "weight", "Source": "H.1 — prior weight"},
        ])
        st.dataframe(param_df, hide_index=True, use_container_width=True)

    # ── Architecture ────────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Section A — Distributed Service Architecture")

        services = [
            {"Service": "Data Ingestion", "Runtime": "Python 3.11, asyncio", "Protocol": "WebSocket (Sportradar) + REST", "Latency Budget": "≤ 200ms"},
            {"Service": "Model Compute", "Runtime": "Python 3.11, NumPy 1.26, SciPy 1.12", "Protocol": "Redis Streams consumer", "Latency Budget": "≤ 800ms pre-game, ≤ 400ms live"},
            {"Service": "Execution", "Runtime": "Python 3.11, aiohttp 3.9", "Protocol": "REST (bookmaker APIs)", "Latency Budget": "≤ 150ms"},
            {"Service": "Calibration & Learning", "Runtime": "Python 3.11, APScheduler 3.10", "Protocol": "PostgreSQL reads + writes", "Latency Budget": "Nightly / within 30min of buzzer"},
            {"Service": "Monitoring & Drift", "Runtime": "Python 3.11, prometheus-client 0.20", "Protocol": "Prometheus :9090, Slack webhook", "Latency Budget": "Continuous"},
        ]
        st.dataframe(pd.DataFrame(services), hide_index=True, use_container_width=True)

        st.markdown("**Data Stores:**")
        stores = [
            {"Store": "PostgreSQL 15", "Library": "psycopg2 2.9 / asyncpg 0.29", "Contents": "Game states, bets, P&L, calibration params, risk events"},
            {"Store": "Redis 7.2", "Library": "redis-py 5.0", "Contents": "Live event queue (Streams), sim cache (TTL=60s), session state"},
            {"Store": "S3-compatible (B2/AWS)", "Library": "boto3", "Contents": "Model checkpoints, simulation archives"},
        ]
        st.dataframe(pd.DataFrame(stores), hide_index=True, use_container_width=True)

        st.markdown("**Async Event Pipeline (10 steps):**")
        pipeline = [
            "1. [DATA_INGESTION] Sportradar WebSocket → GameStateSchema → Redis XADD 'game:events'",
            "2. [MODEL_TRIGGER] Consumer group 'model_workers' reads stream → trigger simulation",
            "3. [SIMULATION] SimulationInputSchema → Monte Carlo (E) → SimulationOutputSchema → Redis cache TTL=60s",
            "4. [MARKET_QUERY] Odds API async fetch (aiohttp, timeout=5s) → Shin model → BetSignalSchema",
            "5. [CALIBRATION] p_blend → isotonic → p_final",
            "6. [GATE_CHECK] All 10 gates (L) → if any fail: log to rejected_bets and return",
            "7. [PORTFOLIO_OPTIMISE] SLSQP (I.2) → portfolio-adjusted stake",
            "8. [EXECUTION] PlacedBetSchema → aiohttp POST to bookmaker → fill_confirmed",
            "9. [PERSISTENCE] PlacedBetSchema → asyncpg → PostgreSQL placed_bets table",
            "10. [MONITORING] Prometheus counters: bets_placed_total, bets_rejected_total, gates_failed_by_type, ev_exec_histogram, kelly_fraction_histogram",
        ]
        for step in pipeline:
            st.write(step)

    # ── Pydantic Schemas ────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Pydantic v2 Schema Reference")
        st.caption("All inter-service schemas use ConfigDict(strict=True)")

        schemas = {
            "GameStateSchema": "score_home ≥ 0, score_away ≥ 0, time_remaining ∈ [0,720], period ∈ {1..5}, fatigue ∈ [0,1], pace_state ∈ [0.7,1.3], market_state ∈ enumerated set",
            "SimulationInputSchema": "lineup_home/away: exactly 5 int player IDs, pace ∈ [80,115], N ∈ {1000,5000,10000,25000,100000}",
            "SimulationOutputSchema": "5 probability outputs, all clipped to [0.001, 0.999]",
            "BetSignalSchema": "p_final/p_market ∈ [0,1], ev_exec, kelly_fraction ∈ [0,0.35], stake > 0, market_type ∈ enumerated, all_gates_passed bool, failed_gates list[str]",
            "PlacedBetSchema": "extends BetSignalSchema + bet_id UUID, placed_at datetime, odds_at_placement, fill_confirmed, fill_timestamp",
            "ResolvedBetSchema": "extends PlacedBetSchema + outcome bool, pnl float, clv float, brier_contribution float",
            "DCParameterSchema": "team_id int, alpha ∈ [0.5,2.0], beta ∈ [0.5,2.0], last_updated datetime, game_count int",
            "RiskEventSchema": "event_type ∈ {STOP_7D_DRAWDOWN, STOP_DAILY_LOSS, ...8 types}, triggered_at, resolved_at, details dict",
        }
        for schema_name, fields in schemas.items():
            with st.expander(f"**{schema_name}**"):
                st.write(fields)
                st.caption("model_config = ConfigDict(strict=True)")
