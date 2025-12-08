"""Streamlit dashboard for PromptSearch experiment visualization."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from promptsearch.logger import SQLiteLogger

# --- Theme and Styling ---
# LangSmith-inspired dark theme with orange accents
COLORS = {
    "bg_dark": "#0E1117",
    "bg_card": "#1E2128",
    "bg_input": "#262B36",
    "accent_orange": "#FF6B35",
    "accent_orange_light": "#FF8C5A",
    "accent_orange_dark": "#CC4D1A",
    "text_primary": "#FAFAFA",
    "text_secondary": "#A0A0A0",
    "success": "#00C853",
    "warning": "#FFD600",
    "error": "#FF5252",
    "border": "#333844",
}

CUSTOM_CSS = f"""
<style>
    /* Main app styling */
    .stApp {{
        background-color: {COLORS['bg_dark']};
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {COLORS['bg_card']};
        border-right: 1px solid {COLORS['border']};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {COLORS['text_primary']} !important;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }}
    
    .kpi-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {COLORS['accent_orange']};
        margin: 0;
    }}
    
    .kpi-label {{
        font-size: 0.9rem;
        color: {COLORS['text_secondary']};
        margin-top: 5px;
    }}
    
    /* Code blocks */
    .prompt-box {{
        background-color: {COLORS['bg_input']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 15px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.85rem;
        color: {COLORS['text_primary']};
        white-space: pre-wrap;
        word-wrap: break-word;
        max-height: 300px;
        overflow-y: auto;
    }}
    
    /* Diff styling */
    .diff-added {{
        background-color: rgba(0, 200, 83, 0.15);
        border-left: 3px solid {COLORS['success']};
        padding-left: 10px;
    }}
    
    .diff-removed {{
        background-color: rgba(255, 82, 82, 0.15);
        border-left: 3px solid {COLORS['error']};
        padding-left: 10px;
    }}
    
    /* Section headers */
    .section-header {{
        border-bottom: 2px solid {COLORS['accent_orange']};
        padding-bottom: 8px;
        margin-bottom: 20px;
    }}
    
    /* Metric override */
    [data-testid="stMetric"] {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 15px;
    }}
    
    [data-testid="stMetricValue"] {{
        color: {COLORS['accent_orange']} !important;
    }}
    
    /* Selectbox styling */
    .stSelectbox > div > div {{
        background-color: {COLORS['bg_input']};
    }}
    
    /* Slider styling */
    .stSlider > div > div > div {{
        background-color: {COLORS['accent_orange']} !important;
    }}
</style>
"""


def format_timestamp(ts: str) -> str:
    """Format ISO timestamp to human-readable."""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return ts or "N/A"


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    return text[:max_length] + "..." if len(text) > max_length else text


def get_db_path() -> str:
    """Get the database path from query params or default."""
    params = st.query_params
    return params.get("db", "promptsearch.db")


def load_experiments(logger: SQLiteLogger) -> list:
    """Load all experiments from the database."""
    return logger.get_experiments()


def main():
    """Main dashboard entry point."""
    st.set_page_config(
        page_title="PromptSearch Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize database connection
    db_path = get_db_path()
    
    if not Path(db_path).exists():
        st.error(f"Database not found: `{db_path}`")
        st.info("Run an optimization experiment first to generate data, then refresh.")
        st.code("from promptsearch import PromptSearcher\n# ... run .optimize()")
        return

    logger = SQLiteLogger(db_path=db_path)
    experiments = load_experiments(logger)

    if not experiments:
        st.warning("No experiments found in the database.")
        st.info("Run an optimization experiment to see data here.")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"""
            <div style="text-align: center; padding: 20px 0;">
                <h1 style="color: {COLORS['accent_orange']}; margin: 0;">🔍 PromptSearch</h1>
                <p style="color: {COLORS['text_secondary']}; font-size: 0.85rem;">Experiment Dashboard</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Experiment selector
        st.subheader("📊 Select Experiment")
        
        experiment_options = {}
        for exp in experiments:
            timestamp = format_timestamp(exp.get("started_at", ""))
            target_preview = truncate_text(exp.get("target_output", ""), 30)
            label = f"{timestamp} | {target_preview}"
            experiment_options[label] = exp["id"]
        
        selected_label = st.selectbox(
            "Experiment",
            options=list(experiment_options.keys()),
            label_visibility="collapsed",
        )
        
        selected_experiment_id = experiment_options.get(selected_label)
        
        st.markdown("---")
        
        # Database info
        st.markdown(f"""
            <div style="font-size: 0.75rem; color: {COLORS['text_secondary']};">
                <strong>Database:</strong> {db_path}<br>
                <strong>Experiments:</strong> {len(experiments)}
            </div>
        """, unsafe_allow_html=True)

    # --- Main Content ---
    if not selected_experiment_id:
        st.info("Select an experiment from the sidebar.")
        return

    experiment = logger.get_experiment(selected_experiment_id)
    generations_data = logger.get_generations(selected_experiment_id)

    if not experiment:
        st.error("Experiment not found.")
        return

    # Header
    st.markdown(f"""
        <h1 class="section-header" style="color: {COLORS['text_primary']};">
            Experiment Analysis
        </h1>
    """, unsafe_allow_html=True)

    # --- KPI Row ---
    col1, col2, col3, col4 = st.columns(4)
    
    best_score = experiment.get("best_score", 0) or 0
    total_gens = experiment.get("generations_planned", 0) or 0
    started = experiment.get("started_at")
    completed = experiment.get("completed_at")
    
    # Calculate duration
    duration_str = "N/A"
    if started and completed:
        try:
            start_dt = datetime.fromisoformat(started)
            end_dt = datetime.fromisoformat(completed)
            duration = (end_dt - start_dt).total_seconds()
            if duration < 60:
                duration_str = f"{duration:.1f}s"
            else:
                duration_str = f"{duration / 60:.1f}m"
        except ValueError:
            pass

    with col1:
        st.metric("🎯 Best Score", f"{best_score:.3f}")
    
    with col2:
        st.metric("🔄 Generations", total_gens)
    
    with col3:
        st.metric("⏱️ Duration", duration_str)
    
    with col4:
        st.metric("🤖 Model", experiment.get("model", "N/A"))

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Score Evolution Chart ---
    if generations_data:
        st.markdown(f"""
            <h2 style="color: {COLORS['text_primary']};">📈 Score Evolution</h2>
        """, unsafe_allow_html=True)

        # Build DataFrame
        df = pd.DataFrame(generations_data)
        
        # Aggregate by step (avg score per generation)
        summary = df.groupby("step_num").agg({
            "score": ["mean", "max", "min", "std"],
            "prompt_text": "first",
            "output_text": "first",
        }).reset_index()
        summary.columns = ["step_num", "avg_score", "max_score", "min_score", "std_score", "prompt_text", "output_text"]
        summary["std_score"] = summary["std_score"].fillna(0)

        # Create Plotly figure
        fig = go.Figure()

        # Add score line
        fig.add_trace(go.Scatter(
            x=summary["step_num"],
            y=summary["avg_score"],
            mode="lines+markers",
            name="Avg Score",
            line=dict(color=COLORS["accent_orange"], width=3),
            marker=dict(size=12, color=COLORS["accent_orange"], line=dict(width=2, color=COLORS["text_primary"])),
            hovertemplate=(
                "<b>Generation %{x}</b><br>"
                "Avg Score: %{y:.4f}<br>"
                "<extra></extra>"
            ),
        ))

        # Add max score line
        fig.add_trace(go.Scatter(
            x=summary["step_num"],
            y=summary["max_score"],
            mode="lines",
            name="Max Score",
            line=dict(color=COLORS["success"], width=1, dash="dash"),
            hovertemplate="Max: %{y:.4f}<extra></extra>",
        ))

        # Add min score line
        fig.add_trace(go.Scatter(
            x=summary["step_num"],
            y=summary["min_score"],
            mode="lines",
            name="Min Score",
            line=dict(color=COLORS["error"], width=1, dash="dash"),
            hovertemplate="Min: %{y:.4f}<extra></extra>",
        ))

        # Style the chart
        fig.update_layout(
            plot_bgcolor=COLORS["bg_card"],
            paper_bgcolor=COLORS["bg_dark"],
            font=dict(color=COLORS["text_primary"]),
            xaxis=dict(
                title="Generation #",
                gridcolor=COLORS["border"],
                tickmode="linear",
                dtick=1,
            ),
            yaxis=dict(
                title="Score (0.0 - 1.0)",
                gridcolor=COLORS["border"],
                range=[0, 1.05],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- Generation Selector ---
        st.markdown(f"""
            <h2 style="color: {COLORS['text_primary']};">🔬 Deep Dive</h2>
        """, unsafe_allow_html=True)

        max_step = int(summary["step_num"].max())
        selected_step = st.slider(
            "Select Generation",
            min_value=0,
            max_value=max_step,
            value=max_step,
            step=1,
        )

        # Get data for selected step
        step_data = df[df["step_num"] == selected_step].iloc[0] if len(df[df["step_num"] == selected_step]) > 0 else None

        if step_data is not None:
            step_score = summary[summary["step_num"] == selected_step]["avg_score"].values[0]
            
            # Info bar
            st.markdown(f"""
                <div style="background-color: {COLORS['bg_card']}; border-left: 4px solid {COLORS['accent_orange']}; 
                     padding: 15px; border-radius: 4px; margin-bottom: 20px;">
                    <strong style="color: {COLORS['accent_orange']};">Generation {selected_step}</strong> | 
                    Score: <strong>{step_score:.4f}</strong>
                </div>
            """, unsafe_allow_html=True)

            # Two-column layout for prompt and outputs
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown(f"**🎛️ System Prompt**")
                st.markdown(f"""
                    <div class="prompt-box">{step_data['prompt_text']}</div>
                """, unsafe_allow_html=True)

            with col_right:
                st.markdown(f"**📤 Model Output**")
                st.markdown(f"""
                    <div class="prompt-box">{step_data['output_text']}</div>
                """, unsafe_allow_html=True)

            # Target output
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**🎯 Target Output**")
            target_text = step_data.get("target_text", experiment.get("target_output", "N/A"))
            st.markdown(f"""
                <div class="prompt-box" style="border-color: {COLORS['success']};">{target_text}</div>
            """, unsafe_allow_html=True)

            # --- Prompt Diff (compare to previous generation) ---
            if selected_step > 0:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**📝 Prompt Changes** (from Generation {selected_step - 1})")
                
                prev_step_data = df[df["step_num"] == selected_step - 1]
                if len(prev_step_data) > 0:
                    prev_prompt = prev_step_data.iloc[0]["prompt_text"]
                    curr_prompt = step_data["prompt_text"]
                    
                    if prev_prompt != curr_prompt:
                        # Simple diff display
                        col_prev, col_curr = st.columns(2)
                        with col_prev:
                            st.markdown(f"""
                                <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; margin-bottom: 5px;">
                                    Previous (Gen {selected_step - 1})
                                </div>
                                <div class="prompt-box diff-removed">{prev_prompt}</div>
                            """, unsafe_allow_html=True)
                        with col_curr:
                            st.markdown(f"""
                                <div style="font-size: 0.8rem; color: {COLORS['text_secondary']}; margin-bottom: 5px;">
                                    Current (Gen {selected_step})
                                </div>
                                <div class="prompt-box diff-added">{curr_prompt}</div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Prompt unchanged from previous generation.")

        # --- All Samples Table ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 All Generations Data", expanded=False):
            display_df = df[["step_num", "score", "input_text", "output_text", "timestamp"]].copy()
            display_df["input_text"] = display_df["input_text"].apply(lambda x: truncate_text(str(x), 60))
            display_df["output_text"] = display_df["output_text"].apply(lambda x: truncate_text(str(x), 60))
            display_df.columns = ["Generation", "Score", "Input (preview)", "Output (preview)", "Timestamp"]
            st.dataframe(display_df, use_container_width=True)

    else:
        st.warning("No generation data found for this experiment.")

    # --- Footer ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: center; padding: 20px; color: {COLORS['text_secondary']}; font-size: 0.75rem;">
            PromptSearch Dashboard v0.1.0 | 
            <a href="https://github.com/kiranmusze/promptsearch" style="color: {COLORS['accent_orange']};">GitHub</a>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

