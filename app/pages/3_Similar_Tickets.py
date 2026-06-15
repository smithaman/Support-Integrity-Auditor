"""
Page 3 — Similar Tickets (FAISS Retrieval)
Shows top-K semantically similar historical tickets
with their assigned priorities and similarity scores.
Supports the mismatch argument with comparative evidence.
"""
# TODO: implement
# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  app/pages/3_Similar_Tickets.py
# ─────────────────────────────────────────

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title = "Similar Tickets — SIA",
    page_icon  = "🔗",
    layout     = "wide",
)


def inject_css() -> None:
    st.markdown("""
    <style>
        .similar-card   { background:#1a1a2e; border:1px solid #2a2a3e;
                          border-radius:10px; padding:1rem 1.2rem;
                          margin-bottom:0.7rem; }
        .sim-score      { font-size:1.3rem; font-weight:700; color:#818cf8; }
        .badge-low      { background:#22c55e22; color:#22c55e;
                          border:1px solid #22c55e; padding:2px 8px;
                          border-radius:8px; font-size:0.78rem; }
        .badge-medium   { background:#f59e0b22; color:#f59e0b;
                          border:1px solid #f59e0b; padding:2px 8px;
                          border-radius:8px; font-size:0.78rem; }
        .badge-high     { background:#ef444422; color:#ef4444;
                          border:1px solid #ef4444; padding:2px 8px;
                          border-radius:8px; font-size:0.78rem; }
        .badge-critical { background:#7c3aed22; color:#a855f7;
                          border:1px solid #a855f7; padding:2px 8px;
                          border-radius:8px; font-size:0.78rem; }
    </style>
    """, unsafe_allow_html=True)


def priority_badge(p: str) -> str:
    return f"<span class='badge-{p.lower()}'>{p}</span>"


def render_similar_tickets(similar_tickets: list, query_text: str = "") -> None:
    """Renders FAISS similar ticket results."""

    if not similar_tickets:
        st.info("No similar tickets found.")
        return

    st.markdown(f"**Found {len(similar_tickets)} similar tickets**")

    if query_text:
        st.markdown(f"""
        <div style='background:#1a1a2e; border-radius:8px;
                    padding:0.8rem 1rem; margin-bottom:1rem;
                    font-size:0.85rem; color:#94a3b8;'>
            🔍 Query: <i>"{query_text[:100]}..."</i>
        </div>
        """, unsafe_allow_html=True)

    for ticket in similar_tickets:
        rank     = ticket.get("rank",      0)
        tid      = ticket.get("ticket_id", "")
        subject  = ticket.get("subject",   "No subject")
        priority = ticket.get("priority",  "")
        category = ticket.get("category",  "")
        channel  = ticket.get("channel",   "")
        tier     = ticket.get("tier",      "")
        rt       = ticket.get("resolution_time", 0)
        sim      = ticket.get("similarity", 0)
        mtype    = ticket.get("mismatch_type", "")
        inf_sev  = ticket.get("inferred_severity", "")

        # Similarity color
        if sim >= 0.85:
            sim_color = "#22c55e"
        elif sim >= 0.70:
            sim_color = "#f59e0b"
        else:
            sim_color = "#64748b"

        st.markdown(f"""
        <div class='similar-card'>
            <div style='display:flex; justify-content:space-between;
                        align-items:flex-start;'>
                <div style='flex:1;'>
                    <div style='display:flex; align-items:center; gap:0.5rem;'>
                        <span style='color:#64748b; font-size:0.78rem;'>
                            #{rank}</span>
                        <span style='font-weight:600; color:#e2e8f0;'>
                            {subject[:80]}</span>
                    </div>
                    <div style='margin-top:0.5rem; font-size:0.8rem;
                                color:#64748b; display:flex; gap:1rem;'>
                        <span>🎫 {tid}</span>
                        <span>📂 {category}</span>
                        <span>📡 {channel}</span>
                        <span>👤 {tier}</span>
                        <span>⏱️ {rt:.0f}hrs</span>
                    </div>
                    <div style='margin-top:0.5rem; display:flex;
                                gap:0.5rem; align-items:center;'>
                        {priority_badge(priority) if priority else ''}
                        {f"→ {priority_badge(inf_sev)}" if inf_sev and inf_sev != priority else ''}
                        {f"<span style='color:#94a3b8; font-size:0.78rem;'>({mtype})</span>" if mtype and mtype != "Consistent" else ''}
                    </div>
                </div>
                <div style='text-align:right; min-width:80px;'>
                    <div style='font-size:0.72rem; color:#64748b;'>
                        SIMILARITY</div>
                    <div class='sim-score' style='color:{sim_color};'>
                        {sim:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main() -> None:
    inject_css()

    st.markdown("### 🔗 Similar Tickets")
    st.markdown(
        "Find semantically similar historical tickets "
        "using FAISS vector search."
    )

    # Check resources
    resources = st.session_state.get("resources")
    if resources is None:
        st.error("Models not loaded. Run training pipeline first.")
        return

    if resources.searcher is None:
        st.warning(
            "FAISS index not loaded. "
            "Run `python build_faiss.py` to build the index."
        )
        return

    # ── Source selector ───────────────────────────────────────
    source = st.radio(
        "Search Source",
        options    = ["From last analysis", "Enter new text"],
        horizontal = True,
    )

    query_text   = ""
    similar_tickets = []

    if source == "From last analysis":
        result = st.session_state.get("last_result")
        if result:
            similar_tickets = result.get("similar_tickets", [])
            ticket          = st.session_state.get("last_ticket", {})
            query_text      = (
                ticket.get("Ticket_Subject", "") + " " +
                ticket.get("Ticket_Description", "")
            )
            if not similar_tickets:
                st.info(
                    "No similar tickets from last analysis. "
                    "Try entering new text below."
                )
        else:
            st.info(
                "No recent analysis found. "
                "Analyze a ticket on the Single Ticket page first."
            )

    elif source == "Enter new text":
        col1, col2 = st.columns([3, 1])
        with col1:
            query_text = st.text_area(
                "Enter ticket text to search",
                placeholder = "Describe the ticket issue...",
                height      = 100,
            )
        with col2:
            top_k = st.number_input(
                "Top K results",
                min_value = 1,
                max_value = 20,
                value     = 5,
            )

        if st.button("🔍 Search Similar Tickets", type="primary"):
            if query_text.strip():
                with st.spinner("Searching..."):
                    similar_tickets = resources.searcher.search_by_text(
                        text  = query_text,
                        model = resources.emb_model,
                        k     = top_k,
                    )
            else:
                st.warning("Please enter some text to search.")

    # ── Render results ────────────────────────────────────────
    if similar_tickets:
        st.divider()

        # Priority distribution chart
        st.markdown("**Priority Distribution in Similar Tickets**")
        priority_counts = {}
        for t in similar_tickets:
            p = t.get("priority", "Unknown")
            priority_counts[p] = priority_counts.get(p, 0) + 1

        cols = st.columns(len(priority_counts))
        for i, (p, count) in enumerate(priority_counts.items()):
            with cols[i]:
                st.metric(p, count)

        st.divider()
        render_similar_tickets(similar_tickets, query_text)


if __name__ == "__main__":
    main()