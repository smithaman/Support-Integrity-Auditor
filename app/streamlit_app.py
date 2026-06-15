"""
streamlit_app.py — SIA Streamlit Application Entry Point

Pages:
  1. Single Ticket Analysis  — input one ticket, get prediction + confidence
  2. Evidence Dossier        — explain WHY the ticket was flagged
  3. Similar Tickets         — FAISS retrieval results
  4. Batch CSV Upload        — analyze multiple tickets, download results

Run:
  streamlit run app/streamlit_app.py
"""

# TODO: implement

# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  app/streamlit_app.py — Main Entry Point
#
#  Run with:
#    streamlit run app/streamlit_app.py
# ─────────────────────────────────────────

import sys
from pathlib import Path

# ── Add project root to path ──────────────────────────────────
# Required so all src/ imports work correctly
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title     = "SIA — Support Integrity Auditor",
    page_icon      = "🔍",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)


# ══════════════════════════════════════════════════════════════
#  RESOURCE LOADER (cached across pages)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading SIA models...")
def load_resources():
    """
    Loads all inference resources once and caches them.
    Shared across all Streamlit pages via session state.

    Returns:
        InferenceResources instance (loaded)
    """
    from src.pipeline.inference_pipeline import InferenceResources

    resources = InferenceResources()
    resources.load(config_path="config/config.yaml")
    return resources


# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════

def inject_css() -> None:
    """Injects custom CSS for consistent styling across pages."""
    st.markdown("""
    <style>
        /* ── Main layout ── */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background-color: #0f0f1a;
            border-right: 1px solid #2a2a3e;
        }
        [data-testid="stSidebar"] .stMarkdown {
            color: #c8c8d8;
        }

        /* ── Priority color badges ── */
        .badge-low {
            background: #22c55e22;
            color: #22c55e;
            border: 1px solid #22c55e;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-medium {
            background: #f59e0b22;
            color: #f59e0b;
            border: 1px solid #f59e0b;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-high {
            background: #ef444422;
            color: #ef4444;
            border: 1px solid #ef4444;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge-critical {
            background: #7c3aed22;
            color: #a855f7;
            border: 1px solid #a855f7;
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }

        /* ── Mismatch type badges ── */
        .badge-hidden-crisis {
            background: #ef444422;
            color: #ef4444;
            border: 1px solid #ef4444;
            padding: 4px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .badge-false-alarm {
            background: #f59e0b22;
            color: #f59e0b;
            border: 1px solid #f59e0b;
            padding: 4px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .badge-consistent {
            background: #22c55e22;
            color: #22c55e;
            border: 1px solid #22c55e;
            padding: 4px 14px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 700;
        }

        /* ── Metric cards ── */
        .metric-card {
            background: #1a1a2e;
            border: 1px solid #2a2a3e;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        }
        .metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #818cf8;
        }
        .metric-card .metric-label {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.2rem;
        }

        /* ── Dossier card ── */
        .dossier-card {
            background: #1a1a2e;
            border: 1px solid #2a2a3e;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        /* ── Evidence item ── */
        .evidence-item {
            background: #0f0f1a;
            border-left: 3px solid #818cf8;
            border-radius: 0 8px 8px 0;
            padding: 0.8rem 1rem;
            margin-bottom: 0.6rem;
            font-size: 0.88rem;
        }

        /* ── Confidence bar ── */
        .conf-bar-container {
            background: #2a2a3e;
            border-radius: 6px;
            height: 8px;
            width: 100%;
            margin-top: 0.4rem;
        }
        .conf-bar-fill {
            background: linear-gradient(90deg, #818cf8, #a855f7);
            border-radius: 6px;
            height: 8px;
        }

        /* ── Section headers ── */
        .section-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #818cf8;
            border-bottom: 1px solid #2a2a3e;
            padding-bottom: 0.4rem;
            margin-bottom: 1rem;
        }

        /* ── Similar ticket row ── */
        .similar-ticket {
            background: #1a1a2e;
            border: 1px solid #2a2a3e;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
        }

        /* ── Alert boxes ── */
        .alert-mismatch {
            background: #ef444411;
            border: 1px solid #ef4444;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            color: #ef4444;
            margin-bottom: 1rem;
        }
        .alert-consistent {
            background: #22c55e11;
            border: 1px solid #22c55e;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            color: #22c55e;
            margin-bottom: 1rem;
        }

        /* ── Hide Streamlit default elements ── */
        #MainMenu { visibility: hidden; }
        footer    { visibility: hidden; }
        header    { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar() -> None:
    """Renders the sidebar with navigation and model status."""
    with st.sidebar:

        # Logo / Title
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <div style='font-size: 2.5rem;'>🔍</div>
            <div style='font-size: 1.3rem; font-weight: 700;
                        color: #818cf8; margin-top: 0.5rem;'>
                SIA
            </div>
            <div style='font-size: 0.75rem; color: #64748b;'>
                Support Integrity Auditor
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation info
        st.markdown("**Navigation**")
        st.markdown("""
        <div style='color: #94a3b8; font-size: 0.85rem; line-height: 1.8;'>
        📋 <b>Single Ticket</b> — Analyze one ticket<br>
        📄 <b>Evidence Dossier</b> — View full evidence<br>
        🔗 <b>Similar Tickets</b> — FAISS retrieval<br>
        📦 <b>Batch Upload</b> — Analyze CSV file
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Model status
        st.markdown("**Model Status**")
        try:
            resources = load_resources()
            st.success("Models loaded ✔")

            st.markdown(f"""
            <div style='font-size: 0.78rem; color: #64748b; line-height: 1.8;'>
            🤖 DeBERTa-v3-small<br>
            📐 BGE-small-en-v1.5<br>
            🗂️ FAISS IndexFlatIP<br>
            ⚙️ Device: {str(resources.device)}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Models not loaded ✘")
            st.caption(f"Error: {str(e)[:80]}")
            st.caption(
                "Run `python train_pipeline.py` first "
                "to train the model."
            )

        st.divider()

        # Verification thresholds reminder
        st.markdown("**Verification Thresholds**")
        st.markdown("""
        <div style='font-size: 0.78rem; color: #64748b; line-height: 1.8;'>
        Accuracy  ≥ 83%<br>
        Macro F1  ≥ 0.82<br>
        Recall    ≥ 0.78 (both)
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Links
        st.markdown("**Resources**")
        st.markdown("""
        <div style='font-size: 0.78rem; line-height: 2;'>
        <a href='https://github.com/' style='color: #818cf8;'>
            GitHub Repository
        </a><br>
        <a href='https://www.kaggle.com/datasets/ajverse/customersupport-tickets-crm-dataset'
           style='color: #818cf8;'>
            Dataset (Kaggle)
        </a>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════

def render_home() -> None:
    """Renders the home / landing page."""

    # Hero section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem;'>
        <div style='font-size: 3rem;'>🔍</div>
        <h1 style='font-size: 2.5rem; font-weight: 800;
                   background: linear-gradient(135deg, #818cf8, #a855f7);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   margin: 0.5rem 0;'>
            Support Integrity Auditor
        </h1>
        <p style='color: #94a3b8; font-size: 1.1rem; max-width: 600px;
                  margin: 0 auto;'>
            Semantics-driven, evidence-grounded automated auditor
            that detects Priority Mismatch in CRM support tickets.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # How it works
    st.markdown("### How It Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 1.8rem;'>📥</div>
            <div style='font-weight: 600; margin-top: 0.5rem;'>Input</div>
            <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;'>
                Submit a ticket or batch CSV
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 1.8rem;'>🧠</div>
            <div style='font-weight: 600; margin-top: 0.5rem;'>Analyze</div>
            <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;'>
                BGE embeddings + RT signals fused
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 1.8rem;'>⚖️</div>
            <div style='font-weight: 600; margin-top: 0.5rem;'>Classify</div>
            <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;'>
                DeBERTa-v3-small binary classifier
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size: 1.8rem;'>📋</div>
            <div style='font-weight: 600; margin-top: 0.5rem;'>Explain</div>
            <div style='color: #94a3b8; font-size: 0.8rem; margin-top: 0.3rem;'>
                Evidence dossier + FAISS search
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown("### Pipeline")
    st.code("""
Ticket Text + Metadata
        │
        ├──► Signal 1: BGE Embeddings → Anchor Similarity → Semantic Score (1–4)
        ├──► Signal 2: Resolution Time → Percentile Ranking → RT Score (1–4)
        │
        ▼
Signal Fusion  →  0.7 × Semantic + 0.3 × RT
        │
        ▼
Inferred Severity  →  Compare with Assigned Priority  →  Pseudo Labels
        │
        ▼
DeBERTa-v3-small Classifier  →  Mismatch Prediction + Confidence
        │
        ▼
FAISS Semantic Search  →  Evidence Dossier  →  Hallucination Verification
    """, language="")

    st.divider()

    # Mismatch types
    st.markdown("### Mismatch Types Detected")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='dossier-card'>
            <div style='font-size: 1.5rem;'>🚨</div>
            <div style='font-weight: 700; color: #ef4444;
                        margin-top: 0.5rem;'>Hidden Crisis</div>
            <div style='color: #94a3b8; font-size: 0.9rem;
                        margin-top: 0.5rem;'>
                Ticket assigned <b>Low/Medium</b> but semantics
                indicate <b>High/Critical</b> severity.
                Under-triaged — at risk of SLA breach.
            </div>
            <div style='margin-top: 0.8rem; font-size: 0.8rem;
                        color: #64748b;'>
                Example: "System outage affecting 500 users" labeled Low
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='dossier-card'>
            <div style='font-size: 1.5rem;'>⚠️</div>
            <div style='font-weight: 700; color: #f59e0b;
                        margin-top: 0.5rem;'>False Alarm</div>
            <div style='color: #94a3b8; font-size: 0.9rem;
                        margin-top: 0.5rem;'>
                Ticket assigned <b>High/Critical</b> but semantics
                indicate <b>Low/Medium</b> severity.
                Over-triaged — wastes urgent resources.
            </div>
            <div style='margin-top: 0.8rem; font-size: 0.8rem;
                        color: #64748b;'>
                Example: "How to change profile picture" labeled Critical
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Quick start
    st.markdown("### Quick Start")
    st.markdown("""
    Use the **sidebar navigation** to access the four pages:

    | Page | Description |
    |------|-------------|
    | 📋 Single Ticket Analysis | Enter one ticket manually and get instant analysis |
    | 📄 Evidence Dossier | View detailed structured evidence for any flagged ticket |
    | 🔗 Similar Tickets | Find semantically similar historical tickets via FAISS |
    | 📦 Batch CSV Upload | Upload a CSV file and analyze multiple tickets at once |
    """)

    st.divider()

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #475569;
                font-size: 0.8rem; padding: 1rem 0;'>
        SIA — Support Integrity Auditor &nbsp;|&nbsp;
        MARS Open Projects 2026 &nbsp;|&nbsp;
        Models and Robotics Section
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point for the Streamlit app."""

    # Inject CSS
    inject_css()

    # Render sidebar
    render_sidebar()

    # Store resources in session state for page access
    if "resources" not in st.session_state:
        try:
            st.session_state["resources"] = load_resources()
            st.session_state["resources_loaded"] = True
        except Exception as e:
            st.session_state["resources"] = None
            st.session_state["resources_loaded"] = False
            st.session_state["resources_error"] = str(e)

    # Render home page
    render_home()


if __name__ == "__main__":
    main()
    2.