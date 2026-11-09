"""
final output 3.py - Academic Operations RAG Hallucination Radar & Quality Dashboard
Interactive Streamlit application for reviewing reference-free RAG evaluation benchmarks,
testing threshold sensitivity, inspecting student doubts, and managing faculty escalation queues.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Academic Operations RAG Radar",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 14px;
        color: #4a5568;
        margin-bottom: 15px;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .badge-approved {
        background-color: #def7ec;
        color: #03543f;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-escalated {
        background-color: #fde8e8;
        color: #9b1c1c;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
DATA_PATH = "final output 2.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(os.path.dirname(__file__), "final output 2.csv")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

raw_df = load_data()

# Header
st.markdown('<div class="main-title">🎓 Academic Operations: RAG Hallucination Radar</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Reference-Free Triad Evaluation & Quality Monitoring for Unlabeled Student Doubts</div>', unsafe_allow_html=True)

if raw_df.empty:
    st.error("Audit log dataset 'final output 2.csv' not found. Please execute the evaluation notebook first.")
    st.stop()

# Sidebar Controls
st.sidebar.header("🎛️ Operational Calibration")

# Live Threshold Slider
rqi_cutoff = st.sidebar.slider(
    "RQI Escalation Cutoff Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.72,
    step=0.01,
    help="Queries with RQI score below this threshold are routed to human faculty."
)

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")
subjects = ["All"] + sorted(raw_df["subject"].dropna().unique().tolist())
selected_subject = st.sidebar.selectbox("Subject", subjects)

tiers = ["All"] + sorted(raw_df["difficulty_tier"].dropna().unique().tolist())
selected_tier = st.sidebar.selectbox("Difficulty Tier", tiers)

# Recalculate based on interactive threshold
df = raw_df.copy()
# Strictly enforce calibrated RQI weights (Alpha=0.25, Beta=0.45, Gamma=0.30)
ALPHA, BETA, GAMMA = 0.25, 0.45, 0.30
df["rqi_score"] = (ALPHA * df["context_relevance"] + BETA * df["groundedness"] + GAMMA * df["answer_relevancy"]).round(4)
df["recalc_decision"] = np.where(df["rqi_score"] >= rqi_cutoff, "Auto-Approved", "Escalated to Faculty")

# Dynamically retag primary risk flags based on recalculated decision
def tag_recalc_risk(row):
    if row["recalc_decision"] == "Auto-Approved":
        return "None (High Quality Grounded Answer)"
    elif row["groundedness"] < 0.60:
        return "Hallucination Risk (Unsupported Claims)"
    elif row["context_relevance"] < 0.55:
        return "Retrieval Drift (Out of Knowledge Base)"
    elif row["answer_relevancy"] < 0.60:
        return "Response Misalignment (Non-responsive)"
    else:
        return "Low Composite Confidence"
df["primary_risk_flag"] = df.apply(tag_recalc_risk, axis=1)

# Apply filters
if selected_subject != "All":
    df = df[df["subject"] == selected_subject]
if selected_tier != "All":
    df = df[df["difficulty_tier"] == selected_tier]

# Key KPI Metrics
col1, col2, col3, col4, col5 = st.columns(5)
total_q = len(df)
auto_q = (df["recalc_decision"] == "Auto-Approved").sum()
esc_q = (df["recalc_decision"] == "Escalated to Faculty").sum()
halluc_q = (df["primary_risk_flag"] == "Hallucination Risk (Unsupported Claims)").sum()
mean_rqi = df["rqi_score"].mean()

col1.metric("Incoming Doubts", f"{total_q}")
col2.metric("Auto-Approved Rate", f"{(auto_q/total_q*100) if total_q > 0 else 0:.1f}%")
col3.metric("Faculty Escalated", f"{esc_q}")
col4.metric("Hallucinations Intercepted", f"{halluc_q}")
col5.metric("Average RQI", f"{mean_rqi:.3f}")

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📊 Executive Quality Radar", "🔬 Hallucination & Query Inspector", "📋 Faculty Escalation Queue"])

with tab1:
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("RAG Triad Dimensional Balance")
        means_data = {
            "Dimension": ["Context Relevance", "Groundedness / Faithfulness", "Answer Relevancy", "Composite RQI"],
            "Score": [df["context_relevance"].mean(), df["groundedness"].mean(), df["answer_relevancy"].mean(), df["rqi_score"].mean()]
        }
        fig_bar = px.bar(
            means_data, x="Dimension", y="Score",
            color="Dimension",
            range_y=[0, 1.0],
            text_auto=".3f",
            title="Mean Benchmark Performance by Dimension"
        )
        fig_bar.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_b:
        st.subheader("Operational Routing Decision Ratio")
        dec_counts = df["recalc_decision"].value_counts().reset_index()
        dec_counts.columns = ["Decision", "Count"]
        fig_pie = px.pie(
            dec_counts, values="Count", names="Decision",
            color="Decision",
            color_discrete_map={"Auto-Approved": "#2ca02c", "Escalated to Faculty": "#d62728"},
            hole=0.45,
            title=f"Routing at Cutoff = {rqi_cutoff:.2f}"
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Groundedness vs Context Relevance Decision Map")
    fig_scatter = px.scatter(
        df, x="context_relevance", y="groundedness",
        color="recalc_decision",
        hover_data=["query_id", "subject", "difficulty_tier", "rqi_score"],
        color_discrete_map={"Auto-Approved": "#2ca02c", "Escalated to Faculty": "#d62728"},
        labels={"context_relevance": "Context Relevance", "groundedness": "Groundedness / Faithfulness"},
        title="Safe Zone vs Hallucination Interception Quadrants"
    )
    fig_scatter.add_hline(y=0.60, line_dash="dash", line_color="gray", annotation_text="Groundedness Cutoff (0.60)")
    fig_scatter.add_vline(x=0.55, line_dash="dot", line_color="gray", annotation_text="Context Relevance Cutoff (0.55)")
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    st.subheader("Student Doubt Inspection & Proposition Trace")
    
    st.dataframe(
        df[["query_id", "subject", "difficulty_tier", "rqi_score", "recalc_decision", "primary_risk_flag", "query_text"]],
        use_container_width=True,
        height=280
    )
    
    selected_qid = st.selectbox("Select Query ID to Inspect Full Reasoning & Grounding:", df["query_id"].tolist())
    if selected_qid:
        q_row = df[df["query_id"] == selected_qid].iloc[0]
        st.markdown(f"### Doubt Inspection: **{selected_qid}** ({q_row['subject']} - {q_row['chapter']})")
        
        c_q1, c_q2 = st.columns([1, 1])
        with c_q1:
            st.info(f"**Student Doubt:**\n\n_{q_row['query_text']}_")
            st.metric("Composite RQI Score", f"{q_row['rqi_score']:.4f}")
            status_color = "green" if q_row['recalc_decision'] == "Auto-Approved" else "red"
            st.markdown(f"**Operational Decision:** :{status_color}[**{q_row['recalc_decision']}**]")
            st.markdown(f"**Primary Risk Flag:** `{q_row['primary_risk_flag']}`")
            
        with c_q2:
            st.success(f"**Generated RAG Response:**\n\n{q_row['generated_answer']}")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Context Rel.", f"{q_row['context_relevance']:.3f}")
            col_m2.metric("Groundedness", f"{q_row['groundedness']:.3f}")
            col_m3.metric("Answer Rel.", f"{q_row['answer_relevancy']:.3f}")

with tab3:
    st.subheader("Faculty Escalation Priority Queue")
    esc_df = df[df["recalc_decision"] == "Escalated to Faculty"].sort_values("rqi_score", ascending=True)
    st.write(f"Displaying **{len(esc_df)}** escalated queries requiring human faculty review:")
    
    st.dataframe(
        esc_df[["query_id", "subject", "chapter", "difficulty_tier", "rqi_score", "primary_risk_flag", "query_text"]],
        use_container_width=True,
        height=350
    )
    
    csv_data = esc_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Export Escalation Queue to CSV",
        data=csv_data,
        file_name="faculty_escalation_queue.csv",
        mime="text/csv"
    )
