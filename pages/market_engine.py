"""NEXOM — Market Probability Engine Page (Section F)."""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from engines.market import (
    shin_two_outcome, default_bookmakers, detect_steam,
    OddsSnapshot, compute_clv, compute_portfolio_clv,
    american_to_implied, remove_vig_proportional,
    BookmakerEfficiency,
)


def render():
    st.title("📈 Market Probability Engine")
    st.caption("Section F — Shin Vig Removal, Bookmaker Efficiency, Steam Detection, CLV")

    tabs = st.tabs(["🔢 Shin Model", "📚 Book Efficiency", "💨 Steam Detection", "📊 CLV Tracker"])

    # ── F.1 Shin Model ─────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("F.1 — Shin (1993) Vig Removal")
        st.latex(r"p_i = \frac{\sqrt{z^2 + 4(1-z) q_i^2 / \text{overround}} - z}{2(1-z)}")
        st.caption("Solved via brentq on [0, 0.2]. z = Shin's insider trading parameter.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input: Decimal Odds**")
            d1 = st.number_input("d₁ (Home / Yes)", 1.01, 10.0, 1.91, 0.01, key="shin_d1")
            d2 = st.number_input("d₂ (Away / No)", 1.01, 10.0, 2.02, 0.01, key="shin_d2")
            use_american = st.checkbox("Convert from American odds")
            if use_american:
                am1 = st.number_input("American odds 1", -500, 500, -105, key="am1")
                am2 = st.number_input("American odds 2", -500, 500, -106, key="am2")
                d1 = (100 / abs(am1) + 1.0) if am1 < 0 else (am1 / 100 + 1.0)
                d2 = (100 / abs(am2) + 1.0) if am2 < 0 else (am2 / 100 + 1.0)
                st.info(f"Decimal: d₁={d1:.3f}, d₂={d2:.3f}")

        with col2:
            st.markdown("**Output: True Probabilities**")
            try:
                p1, p2, z = shin_two_outcome(d1, d2)
                q1, q2 = 1 / d1, 1 / d2
                overround = q1 + q2
                p1_prop, p2_prop = remove_vig_proportional(q1, q2)

                st.metric("Overround", f"{overround:.4f}", f"{(overround-1)*100:.2f}% vig")
                st.metric("z (insider parameter)", f"{z:.6f}")
                st.metric("p_true_1 (Shin)", f"{p1:.4f}")
                st.metric("p_true_2 (Shin)", f"{p2:.4f}")
                st.metric("Sum check", f"{p1+p2:.6f}", "✅ = 1.0" if abs(p1+p2-1) < 1e-6 else "⚠️")

                compare_df = pd.DataFrame({
                    "Method": ["Raw implied", "Proportional removal", "Shin model"],
                    "p₁": [f"{q1:.4f}", f"{p1_prop:.4f}", f"{p1:.4f}"],
                    "p₂": [f"{q2:.4f}", f"{p2_prop:.4f}", f"{p2:.4f}"],
                    "Δp₁": ["-", f"{p1_prop-q1:+.4f}", f"{p1-q1:+.4f}"],
                })
                st.dataframe(compare_df, hide_index=True, use_container_width=True)
            except Exception as e:
                st.error(f"Shin computation error: {e}")

        # Multi-outcome extension note
        st.info("**F.1 Multi-outcome:** For player props with 3+ outcomes, general Shin model with iterative fsolve. For binary markets (moneyline), brentq is exact.")

    # ── F.2 Bookmaker Efficiency ────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("F.2 — Bookmaker Efficiency Scoring")
        st.latex(r"\text{eff}_b = 1 - \frac{\text{MAE}_b}{\text{MAE}_{worst}}")

        books = st.session_state.get("bookmaker_efficiency_state", default_bookmakers())

        rows = []
        for name, b in books.items():
            rows.append({
                "Bookmaker": name.title(),
                "MAE (90d)": f"{b.mae_rolling_90d:.4f}",
                "Efficiency Score": f"{b.efficiency_score:.3f}",
                "Tier": f"Tier-{b.tier}",
                "Classification": {1: "🟢 Sharp", 2: "🟡 Mid", 3: "🔴 Soft"}[b.tier],
            })
        df_books = pd.DataFrame(rows)

        # Color by tier
        def color_tier(val):
            if "Sharp" in str(val):
                return "background-color: #1a3a1a"
            if "Soft" in str(val):
                return "background-color: #3a1a1a"
            return "background-color: #2a2a1a"

        st.dataframe(
            df_books.style.applymap(color_tier, subset=["Classification"]),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("""
**Tiers:**
- **Tier 1 (Sharp):** eff_b > 0.75 — Pinnacle, Circa. Used as market anchor.
- **Tier 2 (Mid):** eff_b ∈ [0.40, 0.75] — Bet365, FanDuel, DraftKings
- **Tier 3 (Soft):** eff_b < 0.40 — Accepted only if corroborated by ≥ 2 Tier-1/2 books within 0.5% tolerance

**F.3 Market Filtering:**
- Lines with spread from Pinnacle anchor > 8 cents → flagged for manual review
- Minimum moneyline depth: $50,000 estimated market depth
        """)

        with st.expander("Simulate MAE update"):
            book_sel = st.selectbox("Bookmaker", list(books.keys()))
            new_mae = st.slider("New MAE observation", 0.05, 0.40, 0.22, 0.01)
            if st.button("Apply EMA update (α=0.05)"):
                books[book_sel].update_mae(new_mae, alpha=0.05)
                st.session_state.bookmaker_efficiency_state = books
                st.success(f"Updated {book_sel}: MAE={books[book_sel].mae_rolling_90d:.4f}, eff={books[book_sel].efficiency_score:.3f}, Tier={books[book_sel].tier}")
                st.rerun()

    # ── F.4 Steam Detection ────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("F.4 — Steam Move Detection")
        st.markdown("""
**Detection conditions (ALL must be met):**
1. Pinnacle line moves ≥ 5 cents (decimal ×100) within 90-second window
2. ≥ 3 Tier-1/2 books move same direction simultaneously
3. Cross-market correlation: moneyline & spread move consistently
4. Volume proxy: Pinnacle price degrades (line moves against bettor)
        """)

        if "steam_snapshots" not in st.session_state:
            st.session_state.steam_snapshots = []

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Add Odds Snapshot**")
            snap_book = st.selectbox("Bookmaker", ["pinnacle", "circa", "bet365", "fanduel", "draftkings"], key="snap_book")
            snap_odds = st.number_input("Current decimal odds", 1.5, 3.0, 1.91, 0.01, key="snap_odds")
            snap_seconds_ago = st.slider("Seconds ago", 0, 120, 0, key="snap_sec")
            if st.button("Add snapshot"):
                snap = OddsSnapshot(
                    bookmaker=snap_book,
                    decimal_odds=snap_odds,
                    timestamp=datetime.utcnow() - timedelta(seconds=snap_seconds_ago),
                )
                st.session_state.steam_snapshots.append(snap)
                st.success(f"Added: {snap_book} @ {snap_odds:.3f}")

        with col2:
            st.markdown("**Snapshot Buffer**")
            if st.session_state.steam_snapshots:
                snap_df = pd.DataFrame([
                    {"Book": s.bookmaker, "Odds": s.decimal_odds,
                     "Age (s)": round((datetime.utcnow() - s.timestamp).total_seconds())}
                    for s in st.session_state.steam_snapshots[-10:]
                ])
                st.dataframe(snap_df, hide_index=True, use_container_width=True)
            else:
                st.info("No snapshots yet")

            if st.button("Clear snapshots"):
                st.session_state.steam_snapshots = []
                st.rerun()

        if st.button("🔍 Run Steam Detection", type="primary"):
            result = detect_steam(st.session_state.steam_snapshots)
            if result:
                st.error(f"🚨 STEAM DETECTED — Direction: {result.direction.upper()}, Move: {result.line_delta_cents:.1f} cents, Books: {result.books_confirming}")
                st.warning("Actions: stake ×0.4, min EV raised to 5.5%, limit orders only, 20s window")
                st.session_state.steam_events.append({
                    "detected_at": result.detected_at.isoformat(),
                    "delta_cents": result.line_delta_cents,
                    "books": result.books_confirming,
                    "direction": result.direction,
                })
            else:
                st.success("✅ No steam detected in current snapshot buffer")

        if st.session_state.steam_events:
            st.subheader("Recent Steam Events")
            st.dataframe(pd.DataFrame(st.session_state.steam_events[-5:]), hide_index=True, use_container_width=True)

    # ── F.5 CLV Tracker ────────────────────────────────────────────────────────
    with tabs[3]:
        st.subheader("F.5 — Closing Line Value (CLV)")
        st.latex(r"\text{CLV}_{bet} = \frac{d_{entry}}{d_{close}} - 1")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Single Bet CLV**")
            d_entry = st.number_input("d_entry (decimal odds at bet)", 1.5, 5.0, 1.95, 0.01)
            d_close = st.number_input("d_close (Pinnacle closing line)", 1.5, 5.0, 1.88, 0.01)
            clv_single = compute_clv(d_entry, d_close)
            color = "✅" if clv_single > 0 else "⚠️"
            st.metric(f"{color} CLV", f"{clv_single:.4f}", f"{clv_single*100:.2f}%",
                      delta_color="normal" if clv_single > 0 else "inverse")
            if clv_single > 0:
                st.success("Placed at better-than-closing odds — positive quality signal")
            else:
                st.warning("Market moved against position — negative CLV")

        with col2:
            st.markdown("**Portfolio CLV (30-day)**")
            np.random.seed(7)
            demo_bets = [
                {"clv": float(np.random.normal(0.013, 0.02)), "stake": float(np.random.uniform(50, 500))}
                for _ in range(50)
            ]
            port_clv = compute_portfolio_clv(demo_bets)
            target_ok = port_clv > 0.012
            st.metric(
                "CLV Portfolio 30d",
                f"{port_clv:.4f}",
                delta=f"{'✅ ' if target_ok else '⚠️ '} target > 0.012",
                delta_color="normal" if target_ok else "inverse",
            )
            st.metric("Sample size", f"{len(demo_bets)} bets")

            clv_series = pd.DataFrame({
                "Bet #": range(1, len(demo_bets) + 1),
                "CLV": [b["clv"] for b in demo_bets],
                "Cumulative CLV": np.cumsum([b["clv"] * b["stake"] for b in demo_bets]) /
                                  np.cumsum([b["stake"] for b in demo_bets]),
            })
            st.line_chart(clv_series.set_index("Bet #")["Cumulative CLV"])
