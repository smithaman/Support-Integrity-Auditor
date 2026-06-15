"""
Page 4 — Batch CSV Upload
Upload a CSV of multiple tickets.
Runs full inference pipeline on all tickets.
Displays summary dashboard + allows download of results as CSV/JSON.
"""
# TODO: implement
# ─────────────────────────────────────────
#  SIA — Support Integrity Auditor
#  app/pages/4_Batch_Upload.py
# ─────────────────────────────────────────

import sys
import json
import io
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title = "Batch Upload — SIA",
    page_icon  = "📦",
    layout     = "wide",
)


def inject_css() -> None:
    st.markdown("""
    <style>
        .stat-card  { background:#1a1a2e; border:1px solid #2a2a3e;
                      border-radius:12px; padding:1.2rem;
                      text-align:center; }
        .stat-value { font-size:2rem; font-weight:700; color:#818cf8; }
        .stat-label { font-size:0.78rem; color:#94a3b8; margin-top:0.2rem; }
        .badge-low  { background:#22c55e22; color:#22c55e;
                      border:1px solid #22c55e; padding:1px 7px;
                      border-radius:8px; font-size:0.75rem; }
        .badge-medium { background:#f59e0b22; color:#f59e0b;
                        border:1px solid #f59e0b; padding:1px 7px;
                        border-radius:8px; font-size:0.75rem; }
        .badge-high { background:#ef444422; color:#ef4444;
                      border:1px solid #ef4444; padding:1px 7px;
                      border-radius:8px; font-size:0.75rem; }
        .badge-critical { background:#7c3aed22; color:#a855f7;
                          border:1px solid #a855f7; padding:1px 7px;
                          border-radius:8px; font-size:0.75rem; }
    </style>
    """, unsafe_allow_html=True)


def render_dashboard(df: pd.DataFrame, dossiers: list) -> None:
    """Renders the Priority Mismatch Dashboard."""

    st.markdown("### 📊 Priority Mismatch Dashboard")

    # ── Summary metrics ───────────────────────────────────────
    n_total     = len(df)
    n_mismatch  = int((df["Prediction"] == 1).sum()) \
                  if "Prediction" in df.columns else 0
    n_consistent = n_total - n_mismatch
    mismatch_rate = n_mismatch / n_total if n_total > 0 else 0

    n_hidden = 0
    n_false  = 0
    if "Mismatch_Type" in df.columns:
        n_hidden = int(
            (df["Mismatch_Type"] == "Hidden Crisis").sum()
        )
        n_false  = int(
            (df["Mismatch_Type"] == "False Alarm").sum()
        )

    col1, col2, col3, col4, col5 = st.columns(5)

    metrics = [
        ("Total Tickets",    n_total,        "#818cf8"),
        ("Mismatches",       n_mismatch,     "#ef4444"),
        ("Consistent",       n_consistent,   "#22c55e"),
        ("Hidden Crisis",    n_hidden,       "#ef4444"),
        ("False Alarm",      n_false,        "#f59e0b"),
    ]

    for col, (label, value, color) in zip(
        [col1, col2, col3, col4, col5], metrics
    ):
        with col:
            st.markdown(f"""
            <div class='stat-card'>
                <div class='stat-value' style='color:{color};'>
                    {value}</div>
                <div class='stat-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        # Mismatch distribution pie
        fig = go.Figure(data=[go.Pie(
            labels = ["Consistent", "Hidden Crisis", "False Alarm"],
            values = [n_consistent, n_hidden, n_false],
            hole   = 0.5,
            marker = dict(colors=["#22c55e", "#ef4444", "#f59e0b"]),
        )])
        fig.update_layout(
            title      = "Mismatch Distribution",
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font_color    = "#94a3b8",
            showlegend    = True,
            height        = 280,
            margin        = dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Mismatch by assigned priority
        if "Priority_Level" in df.columns and "Prediction" in df.columns:
            priority_mismatch = (
                df.groupby("Priority_Level")["Prediction"]
                .mean()
                .reset_index()
            )
            priority_mismatch.columns = ["Priority", "Mismatch Rate"]
            priority_mismatch = priority_mismatch.sort_values(
                "Priority",
                key=lambda x: x.map({"Low":0,"Medium":1,"High":2,"Critical":3})
            )

            fig2 = px.bar(
                priority_mismatch,
                x     = "Priority",
                y     = "Mismatch Rate",
                color = "Priority",
                color_discrete_map = {
                    "Low":      "#22c55e",
                    "Medium":   "#f59e0b",
                    "High":     "#ef4444",
                    "Critical": "#a855f7",
                },
                title = "Mismatch Rate by Priority",
            )
            fig2.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                font_color    = "#94a3b8",
                showlegend    = False,
                height        = 280,
                margin        = dict(t=40, b=0, l=0, r=0),
            )
            fig2.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig2, use_container_width=True)

    with col3:
        # Confidence distribution
        if "Confidence" in df.columns:
            fig3 = px.histogram(
                df[df["Prediction"] == 1],
                x       = "Confidence",
                nbins   = 20,
                title   = "Confidence Distribution (Mismatches)",
                color_discrete_sequence = ["#818cf8"],
            )
            fig3.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                plot_bgcolor  = "rgba(0,0,0,0)",
                font_color    = "#94a3b8",
                height        = 280,
                margin        = dict(t=40, b=0, l=0, r=0),
                showlegend    = False,
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── Severity delta heatmap ────────────────────────────────
    if (
        "Issue_Category" in df.columns and
        "Ticket_Channel" in df.columns and
        "Severity_Delta_Abs" in df.columns
    ):
        st.markdown("**Severity Delta Heatmap — Category × Channel**")

        heatmap_data = (
            df.groupby(["Issue_Category", "Ticket_Channel"])
            ["Severity_Delta_Abs"]
            .mean()
            .reset_index()
            .pivot(
                index   = "Issue_Category",
                columns = "Ticket_Channel",
                values  = "Severity_Delta_Abs",
            )
        )

        fig4 = px.imshow(
            heatmap_data,
            color_continuous_scale = "Purples",
            title  = "Mean Severity Delta by Category and Channel",
            aspect = "auto",
        )
        fig4.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font_color    = "#94a3b8",
            height        = 300,
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    # ── Flagged tickets table ─────────────────────────────────
    st.markdown("**Flagged Tickets**")

    flagged = df[df["Prediction"] == 1].copy() \
              if "Prediction" in df.columns else df

    if len(flagged) == 0:
        st.success("No mismatches detected in this batch.")
        return

    # Select display columns
    display_cols = [
        c for c in [
            "Ticket_ID", "Ticket_Subject",
            "Priority_Level", "Inferred_Severity",
            "Mismatch_Type", "Confidence",
            "Severity_Delta_Abs",
        ] if c in flagged.columns
    ]

    if display_cols:
        display_df = flagged[display_cols].copy()

        if "Confidence" in display_df.columns:
            display_df["Confidence"] = display_df["Confidence"].round(3)
        if "Severity_Delta_Abs" in display_df.columns:
            display_df["Severity_Delta_Abs"] = (
                display_df["Severity_Delta_Abs"].round(2)
            )

        st.dataframe(
            display_df,
            use_container_width = True,
            height              = 350,
        )

    # ── Download buttons ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        csv_all = df.to_csv(index=False).encode()
        st.download_button(
            label     = "⬇ Download All Predictions CSV",
            data      = csv_all,
            file_name = "sia_predictions.csv",
            mime      = "text/csv",
            use_container_width = True,
        )

    with col2:
        csv_flagged = flagged.to_csv(index=False).encode()
        st.download_button(
            label     = "⬇ Download Mismatches Only CSV",
            data      = csv_flagged,
            file_name = "sia_mismatches.csv",
            mime      = "text/csv",
            use_container_width = True,
        )

    with col3:
        if dossiers:
            dossier_json = json.dumps(dossiers, indent=2).encode()
            st.download_button(
                label     = "⬇ Download Dossiers JSON",
                data      = dossier_json,
                file_name = "sia_dossiers.json",
                mime      = "application/json",
                use_container_width = True,
            )


def main() -> None:
    inject_css()

    st.markdown("### 📦 Batch CSV Upload")
    st.markdown(
        "Upload a CSV of support tickets to analyze multiple tickets at once."
    )

    # Check resources
    resources = st.session_state.get("resources")
    if resources is None:
        st.error("Models not loaded. Run training pipeline first.")
        return

    # ── Upload section ────────────────────────────────────────
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Tickets CSV",
            type = ["csv"],
            help = (
                "CSV must contain: Ticket_Subject, Ticket_Description, "
                "Priority_Level, Ticket_Channel, Issue_Category, "
                "Resolution_Time_Hours"
            ),
        )

    with col2:
        st.markdown("**Required Columns**")
        st.markdown("""
        <div style='font-size:0.8rem; color:#94a3b8; line-height:1.8;'>
        ✓ Ticket_Subject<br>
        ✓ Ticket_Description<br>
        ✓ Priority_Level<br>
        ✓ Ticket_Channel<br>
        ✓ Issue_Category<br>
        ✓ Resolution_Time_Hours
        </div>
        """, unsafe_allow_html=True)

        # Sample CSV download
        sample_csv = (
            "Ticket_ID,Ticket_Subject,Ticket_Description,"
            "Priority_Level,Ticket_Channel,Issue_Category,"
            "Resolution_Time_Hours,Customer_Email\n"
            "T001,Cannot login,All users affected since 6am,"
            "Low,Email,Technical,72,user@enterprise.org\n"
            "T002,Update email,Please update my email address,"
            "Critical,Chat,Account,1,user@example.com\n"
        )
        st.download_button(
            label     = "⬇ Sample CSV",
            data      = sample_csv,
            file_name = "sample_tickets.csv",
            mime      = "text/csv",
            use_container_width = True,
        )

    # ── Use demo data ─────────────────────────────────────────
    use_demo = st.checkbox(
        "Use demo_data/sample_batch.csv instead",
        value = False,
    )

    # ── Process button ────────────────────────────────────────
    run_button = st.button(
        "🚀 Analyze Batch",
        type = "primary",
        use_container_width = True,
    )

    if run_button:
        # Load data
        if use_demo:
            demo_path = ROOT / "demo_data" / "sample_batch.csv"
            if not demo_path.exists():
                st.error(
                    f"Demo file not found: {demo_path}\n"
                    "Please upload a CSV file instead."
                )
                return
            df_input = pd.read_csv(demo_path)
            st.info(f"Using demo data: {len(df_input)} tickets")

        elif uploaded_file is not None:
            df_input = pd.read_csv(uploaded_file)
            st.info(f"Uploaded: {uploaded_file.name} — {len(df_input)} tickets")

        else:
            st.warning("Please upload a CSV file or enable demo data.")
            return

        # Preview
        with st.expander("Preview Input Data"):
            st.dataframe(df_input.head(5), use_container_width=True)

        # Run inference
        progress_bar = st.progress(0, text="Starting inference...")

        try:
            progress_bar.progress(20, text="Running pipeline...")

            from src.pipeline.inference_pipeline import infer_batch

            progress_bar.progress(40, text="Encoding tickets...")
            predictions_df, dossiers = infer_batch(
                df          = df_input,
                resources   = resources,
                config_path = "config/config.yaml",
            )

            progress_bar.progress(90, text="Generating dashboard...")

            # Store in session
            st.session_state["batch_predictions"] = predictions_df
            st.session_state["batch_dossiers"]    = dossiers

            progress_bar.progress(100, text="Done!")
            st.success(
                f"Analysis complete — "
                f"{len(predictions_df)} tickets processed, "
                f"{int((predictions_df.get('Prediction',0)==1).sum())} "
                f"mismatches detected"
            )

        except Exception as e:
            progress_bar.empty()
            st.error(f"Inference failed: {e}")
            return

        # ── Dashboard ─────────────────────────────────────────
        st.divider()
        render_dashboard(predictions_df, dossiers)

    # ── Show previous results ─────────────────────────────────
    elif "batch_predictions" in st.session_state:
        st.info("Showing results from previous batch run.")
        st.divider()
        render_dashboard(
            st.session_state["batch_predictions"],
            st.session_state.get("batch_dossiers", []),
        )


if __name__ == "__main__":
    main()