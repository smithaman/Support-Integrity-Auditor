"""
Page 2 — Evidence Dossier
Displays the full structured dossier for a flagged ticket:
  assigned_priority, inferred_severity, mismatch_type,
  severity_delta, feature_evidence, constraint_analysis, confidence
"""
# TODO: implement
# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  app/pages/2_Evidence_Dossier.py
# ─────────────────────────────────────────

import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title = "Evidence Dossier — SIA",
    page_icon  = "📄",
    layout     = "wide",
)


def inject_css() -> None:
    st.markdown("""
    <style>
        .dossier-header { background:#1a1a2e; border:1px solid #2a2a3e;
                          border-radius:12px; padding:1.5rem;
                          margin-bottom:1rem; }
        .evidence-item  { background:#0f0f1a; border-left:3px solid #818cf8;
                          border-radius:0 8px 8px 0; padding:0.8rem 1rem;
                          margin-bottom:0.6rem; font-size:0.88rem; }
        .evidence-keyword { border-left-color:#ef4444; }
        .evidence-rt      { border-left-color:#22c55e; }
        .evidence-faiss   { border-left-color:#f59e0b; }
        .field-row  { display:flex; justify-content:space-between;
                      padding:0.4rem 0; border-bottom:1px solid #1e1e2e;
                      font-size:0.88rem; }
        .field-label { color:#64748b; }
        .field-value { color:#e2e8f0; font-weight:500; }
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


def render_dossier(dossier: dict) -> None:
    """Renders a structured evidence dossier."""

    # ── Header ────────────────────────────────────────────────
    mtype   = dossier.get("mismatch_type", "")
    color   = "#ef4444" if mtype == "Hidden Crisis" else "#f59e0b"
    icon    = "🚨" if mtype == "Hidden Crisis" else "⚠️"

    st.markdown(f"""
    <div class='dossier-header'>
        <div style='display:flex; justify-content:space-between;
                    align-items:center;'>
            <div>
                <div style='font-size:0.75rem; color:#64748b;'>
                    TICKET ID</div>
                <div style='font-size:1.2rem; font-weight:700;
                            color:#e2e8f0;'>
                    {dossier.get('ticket_id', 'N/A')}</div>
            </div>
            <div style='text-align:center;'>
                <span style='font-size:1.5rem;'>{icon}</span>
                <div style='font-size:1rem; font-weight:700;
                            color:{color};'>
                    {mtype}
                </div>
            </div>
            <div style='text-align:right;'>
                <div style='font-size:0.75rem; color:#64748b;'>
                    CONFIDENCE</div>
                <div style='font-size:1.2rem; font-weight:700;
                            color:#818cf8;'>
                    {float(dossier.get("confidence", 0)):.1%}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Priority comparison ───────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        ap = dossier.get("assigned_priority", "")
        st.markdown(f"""
        <div style='text-align:center; background:#1a1a2e;
                    border-radius:10px; padding:1rem;'>
            <div style='font-size:0.72rem; color:#64748b;'>
                ASSIGNED PRIORITY</div>
            <div style='margin-top:0.5rem;'>
                {priority_badge(ap)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        delta = dossier.get("severity_delta", "")
        st.markdown(f"""
        <div style='text-align:center; background:#1a1a2e;
                    border-radius:10px; padding:1rem;'>
            <div style='font-size:0.72rem; color:#64748b;'>
                SEVERITY DELTA</div>
            <div style='font-size:1.4rem; font-weight:700;
                        color:#818cf8; margin-top:0.3rem;'>
                Δ {delta}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        isev = dossier.get("inferred_severity", "")
        st.markdown(f"""
        <div style='text-align:center; background:#1a1a2e;
                    border-radius:10px; padding:1rem;'>
            <div style='font-size:0.72rem; color:#64748b;'>
                INFERRED SEVERITY</div>
            <div style='margin-top:0.5rem;'>
                {priority_badge(isev)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Constraint analysis ───────────────────────────────────
    st.markdown("**📝 Constraint Analysis**")
    st.markdown(f"""
    <div style='background:#1a1a2e; border-radius:10px;
                padding:1rem; color:#cbd5e1; font-size:0.9rem;
                line-height:1.7; border-left:3px solid #818cf8;'>
        {dossier.get('constraint_analysis', 'N/A')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature evidence ──────────────────────────────────────
    st.markdown("**🔎 Feature Evidence**")

    for item in dossier.get("feature_evidence", []):
        signal = item.get("signal", "")

        if signal == "keyword":
            extra_cls = "evidence-keyword"
            icon_e    = "🔑"
            title     = f"Keyword: <b>{item.get('value', '')}</b>"
            detail    = (
                f"Category: {item.get('category', '')} | "
                f"Context: <i>{item.get('context', '')}</i>"
            )

        elif signal == "resolution_time":
            extra_cls = "evidence-rt"
            icon_e    = "⏱️"
            title     = f"Resolution Time: <b>{item.get('value', '')}</b>"
            detail    = item.get("interpretation", "")

        elif signal == "semantic_similarity":
            extra_cls = "evidence-faiss"
            icon_e    = "🔗"
            title     = "Semantic Similarity (FAISS)"
            detail    = item.get("pattern", "")

        else:
            extra_cls = ""
            icon_e    = "📌"
            title     = signal
            detail    = str(item)

        st.markdown(f"""
        <div class='evidence-item {extra_cls}'>
            {icon_e} {title}<br>
            <span style='color:#64748b; font-size:0.82rem;'>
                {detail}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # FAISS similar tickets table
        if signal == "semantic_similarity":
            similar = item.get("similar_tickets", [])
            if similar:
                st.markdown(
                    "<div style='padding-left:1rem;'>",
                    unsafe_allow_html=True
                )
                for t in similar:
                    st.markdown(f"""
                    <div style='background:#1a1a2e; border-radius:6px;
                                padding:0.5rem 0.8rem; margin:0.3rem 0;
                                font-size:0.82rem; color:#94a3b8;'>
                        [{t.get('ticket_id','')}]
                        {priority_badge(t.get('priority',''))}
                        &nbsp; {t.get('subject','')[:60]}
                        &nbsp;
                        <span style='color:#64748b;'>
                            sim={t.get('similarity',0):.3f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Raw JSON ──────────────────────────────────────────────
    with st.expander("📋 Raw Dossier JSON"):
        st.json(dossier)

    # ── Download ──────────────────────────────────────────────
    st.download_button(
        label     = "⬇ Download Dossier JSON",
        data      = json.dumps(dossier, indent=2),
        file_name = f"dossier_{dossier.get('ticket_id', 'unknown')}.json",
        mime      = "application/json",
    )


def main() -> None:
    inject_css()

    st.markdown("### 📄 Evidence Dossier")
    st.markdown(
        "Structured, hallucination-free evidence for each flagged ticket.",
    )

    # ── Source selector ───────────────────────────────────────
    source = st.radio(
        "Dossier Source",
        options     = ["Last analyzed ticket", "Load from file", "Paste JSON"],
        horizontal  = True,
    )

    dossier = None

    if source == "Last analyzed ticket":
        result = st.session_state.get("last_result")
        if result and result.get("dossier"):
            dossier = result["dossier"]
            st.success(
                f"Showing dossier for ticket: "
                f"{dossier.get('ticket_id', 'N/A')}"
            )
        else:
            st.info(
                "No recent analysis found. "
                "Go to **Single Ticket Analysis** and analyze a ticket first."
            )

    elif source == "Load from file":
        uploaded = st.file_uploader(
            "Upload dossiers JSON",
            type = ["json"],
        )
        if uploaded:
            data = json.load(uploaded)
            if isinstance(data, list) and len(data) > 0:
                options = [
                    d.get("ticket_id", f"Ticket {i}")
                    for i, d in enumerate(data)
                ]
                selected = st.selectbox("Select ticket", options)
                idx      = options.index(selected)
                dossier  = data[idx]
            elif isinstance(data, dict):
                dossier = data

    elif source == "Paste JSON":
        raw = st.text_area(
            "Paste dossier JSON here",
            height = 200,
        )
        if raw.strip():
            try:
                dossier = json.loads(raw)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # ── Render dossier ────────────────────────────────────────
    if dossier:
        st.divider()
        render_dossier(dossier)


if __name__ == "__main__":
    main()