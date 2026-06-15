"""
Page 1 — Single Ticket Analysis
Form inputs: Subject, Description, Channel, Customer Tier, Resolution Time, Assigned Priority
Outputs: Mismatch / Consistent prediction + confidence score
"""
# TODO: implement
# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  app/pages/1_Single_Ticket_Analysis.py
# ─────────────────────────────────────────

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title = "Single Ticket — SIA",
    page_icon  = "📋",
    layout     = "wide",
)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def priority_badge(priority: str) -> str:
    cls = f"badge-{priority.lower()}"
    return f"<span class='{cls}'>{priority}</span>"


def confidence_bar(conf: float) -> str:
    pct = int(conf * 100)
    return f"""
    <div class='conf-bar-container'>
        <div class='conf-bar-fill' style='width:{pct}%;'></div>
    </div>
    <div style='font-size:0.75rem; color:#64748b;
                margin-top:0.2rem;'>{pct}% confidence</div>
    """


def inject_css() -> None:
    st.markdown("""
    <style>
        .badge-low      { background:#22c55e22; color:#22c55e;
                          border:1px solid #22c55e; padding:2px 10px;
                          border-radius:12px; font-size:0.8rem;
                          font-weight:600; }
        .badge-medium   { background:#f59e0b22; color:#f59e0b;
                          border:1px solid #f59e0b; padding:2px 10px;
                          border-radius:12px; font-size:0.8rem;
                          font-weight:600; }
        .badge-high     { background:#ef444422; color:#ef4444;
                          border:1px solid #ef4444; padding:2px 10px;
                          border-radius:12px; font-size:0.8rem;
                          font-weight:600; }
        .badge-critical { background:#7c3aed22; color:#a855f7;
                          border:1px solid #a855f7; padding:2px 10px;
                          border-radius:12px; font-size:0.8rem;
                          font-weight:600; }
        .result-card    { background:#1a1a2e; border:1px solid #2a2a3e;
                          border-radius:12px; padding:1.5rem;
                          margin-bottom:1rem; }
        .conf-bar-container { background:#2a2a3e; border-radius:6px;
                              height:8px; width:100%;
                              margin-top:0.4rem; }
        .conf-bar-fill  { background:linear-gradient(90deg,#818cf8,#a855f7);
                          border-radius:6px; height:8px; }
        .signal-row     { background:#0f0f1a; border-radius:8px;
                          padding:0.6rem 1rem; margin-bottom:0.4rem;
                          font-size:0.88rem; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════

def render_input_form() -> dict:
    """Renders the ticket input form and returns form values."""

    st.markdown("### 📋 Single Ticket Analysis")
    st.markdown(
        "Enter ticket details below to check for priority mismatch.",
        unsafe_allow_html=True,
    )

    with st.form("ticket_form"):

        col1, col2 = st.columns([2, 1])

        with col1:
            subject = st.text_input(
                "Ticket Subject *",
                placeholder = "e.g. Cannot login to account",
            )
            description = st.text_area(
                "Ticket Description *",
                placeholder = (
                    "e.g. All users in our organization have been "
                    "unable to access the platform since 6am. "
                    "We have a critical client demo in 2 hours."
                ),
                height = 150,
            )

        with col2:
            priority = st.selectbox(
                "Assigned Priority *",
                options = ["Low", "Medium", "High", "Critical"],
                index   = 0,
            )
            channel = st.selectbox(
                "Ticket Channel",
                options = ["Email", "Chat", "Web Form"],
                index   = 0,
            )
            category = st.selectbox(
                "Issue Category",
                options = [
                    "Technical", "Billing",
                    "Account", "General Inquiry", "Fraud"
                ],
                index = 0,
            )
            resolution_time = st.number_input(
                "Resolution Time (hours)",
                min_value = 0.0,
                max_value = 200.0,
                value     = 24.0,
                step      = 0.5,
            )
            email = st.text_input(
                "Customer Email",
                value = "user@example.com",
                help  = "Used to derive customer tier",
            )

        submitted = st.form_submit_button(
            "🔍 Analyze Ticket",
            use_container_width = True,
            type                = "primary",
        )

    if submitted:
        return {
            "Ticket_Subject":       subject,
            "Ticket_Description":   description,
            "Priority_Level":       priority,
            "Ticket_Channel":       channel,
            "Issue_Category":       category,
            "Resolution_Time_Hours": resolution_time,
            "Customer_Email":       email,
            "Ticket_ID":            "MANUAL-001",
        }
    return {}


# ══════════════════════════════════════════════════════════════
#  RESULT RENDERER
# ══════════════════════════════════════════════════════════════

def render_result(result: dict) -> None:
    """Renders the prediction result."""

    prediction  = result["prediction"]
    label       = result["label"]
    confidence  = result["confidence"]
    assigned    = result["assigned_priority"]
    inferred    = result["inferred_severity"]
    mtype       = result["mismatch_type"]
    sem_score   = result["semantic_score"]
    rt_score    = result["rt_score"]
    fused_score = result["fused_score"]
    delta       = result["severity_delta"]

    st.divider()

    # ── Main verdict ──────────────────────────────────────────
    if prediction == 1:
        if mtype == "Hidden Crisis":
            st.markdown(f"""
            <div style='background:#ef444411; border:2px solid #ef4444;
                        border-radius:12px; padding:1.2rem;
                        text-align:center; margin-bottom:1rem;'>
                <div style='font-size:2rem;'>🚨</div>
                <div style='font-size:1.4rem; font-weight:800;
                            color:#ef4444; margin-top:0.3rem;'>
                    HIDDEN CRISIS DETECTED
                </div>
                <div style='color:#94a3b8; margin-top:0.3rem;'>
                    Ticket is under-triaged — assigned {assigned}
                    but inferred severity is {inferred}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background:#f59e0b11; border:2px solid #f59e0b;
                        border-radius:12px; padding:1.2rem;
                        text-align:center; margin-bottom:1rem;'>
                <div style='font-size:2rem;'>⚠️</div>
                <div style='font-size:1.4rem; font-weight:800;
                            color:#f59e0b; margin-top:0.3rem;'>
                    FALSE ALARM DETECTED
                </div>
                <div style='color:#94a3b8; margin-top:0.3rem;'>
                    Ticket is over-triaged — assigned {assigned}
                    but inferred severity is {inferred}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#22c55e11; border:2px solid #22c55e;
                    border-radius:12px; padding:1.2rem;
                    text-align:center; margin-bottom:1rem;'>
            <div style='font-size:2rem;'>✅</div>
            <div style='font-size:1.4rem; font-weight:800;
                        color:#22c55e; margin-top:0.3rem;'>
                PRIORITY CONSISTENT
            </div>
            <div style='color:#94a3b8; margin-top:0.3rem;'>
                Assigned priority {assigned} matches inferred severity
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='result-card' style='text-align:center;'>
            <div style='font-size:0.75rem; color:#64748b;'>
                ASSIGNED PRIORITY</div>
            <div style='margin-top:0.5rem;'>
                {priority_badge(assigned)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='result-card' style='text-align:center;'>
            <div style='font-size:0.75rem; color:#64748b;'>
                INFERRED SEVERITY</div>
            <div style='margin-top:0.5rem;'>
                {priority_badge(inferred)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='result-card' style='text-align:center;'>
            <div style='font-size:0.75rem; color:#64748b;'>
                SEVERITY DELTA</div>
            <div style='font-size:1.6rem; font-weight:700;
                        color:#818cf8; margin-top:0.3rem;'>
                {delta}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='result-card' style='text-align:center;'>
            <div style='font-size:0.75rem; color:#64748b;'>
                CONFIDENCE</div>
            <div style='font-size:1.6rem; font-weight:700;
                        color:#818cf8; margin-top:0.3rem;'>
                {confidence:.1%}
            </div>
            {confidence_bar(confidence)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Signal scores ─────────────────────────────────────────
    st.markdown("**Signal Scores**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='signal-row'>
            <b>Signal 1 — Semantic</b><br>
            <span style='color:#818cf8; font-size:1.1rem;'>
                {sem_score:.3f}</span>
            <span style='color:#64748b;'> / 4.0</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='signal-row'>
            <b>Signal 2 — Resolution Time</b><br>
            <span style='color:#818cf8; font-size:1.1rem;'>
                {rt_score:.3f}</span>
            <span style='color:#64748b;'> / 4.0</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='signal-row'>
            <b>Fused Score</b><br>
            <span style='color:#a855f7; font-size:1.1rem;'>
                {fused_score:.3f}</span>
            <span style='color:#64748b;'> / 4.0</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Store result in session for dossier page ──────────────
    st.session_state["last_result"] = result
    st.session_state["last_ticket"] = st.session_state.get(
        "_current_ticket", {}
    )

    if prediction == 1 and result.get("dossier"):
        st.info(
            "📄 View the full Evidence Dossier on the "
            "**Evidence Dossier** page →",
            icon = "📋",
        )


# ══════════════════════════════════════════════════════════════
#  DEMO TICKETS
# ══════════════════════════════════════════════════════════════

def render_demo_buttons() -> dict:
    """Renders demo ticket quick-fill buttons."""

    st.markdown("**Quick Demo Tickets**")
    col1, col2, col3 = st.columns(3)

    demo_tickets = {
        "🚨 Hidden Crisis": {
            "Ticket_Subject":       "Minor question about account",
            "Ticket_Description":   (
                "Hi support, just a minor question. "
                "Our entire payment processing system has been down "
                "since 3am. We are losing $50,000 per hour and have "
                "500 enterprise clients unable to transact."
            ),
            "Priority_Level":       "Low",
            "Ticket_Channel":       "Email",
            "Issue_Category":       "Technical",
            "Resolution_Time_Hours": 96.0,
            "Customer_Email":       "user@enterprise.org",
            "Ticket_ID":            "DEMO-HC-001",
        },
        "⚠️ False Alarm": {
            "Ticket_Subject":       "URGENT!! CRITICAL!! Help needed ASAP!!",
            "Ticket_Description":   (
                "This is extremely urgent! Critical issue! "
                "I cannot figure out how to change the font size "
                "in my account settings. Please fix this immediately."
            ),
            "Priority_Level":       "Critical",
            "Ticket_Channel":       "Chat",
            "Issue_Category":       "Account",
            "Resolution_Time_Hours": 1.5,
            "Customer_Email":       "user@example.com",
            "Ticket_ID":            "DEMO-FA-001",
        },
        "✅ Consistent": {
            "Ticket_Subject":       "System completely down for all users",
            "Ticket_Description":   (
                "Our entire platform has been inaccessible since "
                "the maintenance window. All 200 engineers cannot "
                "deploy. Product launch in 4 hours. Database corrupted."
            ),
            "Priority_Level":       "Critical",
            "Ticket_Channel":       "Phone",
            "Issue_Category":       "Technical",
            "Resolution_Time_Hours": 3.0,
            "Customer_Email":       "admin@enterprise.org",
            "Ticket_ID":            "DEMO-CON-001",
        },
    }

    selected = None
    with col1:
        if st.button("🚨 Hidden Crisis", use_container_width=True):
            selected = demo_tickets["🚨 Hidden Crisis"]
    with col2:
        if st.button("⚠️ False Alarm", use_container_width=True):
            selected = demo_tickets["⚠️ False Alarm"]
    with col3:
        if st.button("✅ Consistent", use_container_width=True):
            selected = demo_tickets["✅ Consistent"]

    return selected or {}


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()

    # Check resources
    resources = st.session_state.get("resources")
    if resources is None:
        st.error(
            "Models not loaded. "
            "Run `python train_pipeline.py` first, "
            "then restart the app."
        )
        return

    # Demo buttons
    demo_ticket = render_demo_buttons()
    st.divider()

    # Input form
    ticket = render_input_form()

    # Use demo ticket if selected
    if demo_ticket and not ticket:
        ticket = demo_ticket

    # Run inference
    if ticket:
        if not ticket.get("Ticket_Subject", "").strip():
            st.warning("Please enter a Ticket Subject.")
            return
        if not ticket.get("Ticket_Description", "").strip():
            st.warning("Please enter a Ticket Description.")
            return

        st.session_state["_current_ticket"] = ticket

        with st.spinner("Analyzing ticket..."):
            from src.pipeline.inference_pipeline import infer_single_ticket
            result = infer_single_ticket(
                ticket      = ticket,
                resources   = resources,
                config_path = "config/config.yaml",
            )

        render_result(result)


if __name__ == "__main__":
    main()