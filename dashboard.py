# Streamlit dashboard — run with: streamlit run dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_collector import build_timeseries
from skill_analyzer import (
    load_timeseries,
    compute_skill_growth,
    get_emerging_skills,
    get_declining_skills,
    compute_survival_score,
    forecast_demand,
    recommend_skills,
)
from models import predict_skill_demand, get_ai_exposure_score, compute_yoy_growth
from config import AI_EXPOSURE_SCORES

# page config
st.set_page_config(
    page_title="AI Job Market Analyzer",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .big-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label { font-size: 0.78rem; color: #888; text-transform: uppercase; }
    .kpi-value { font-size: 1.6rem; font-weight: 700; }
    .badge-high   { color: #d63031; font-weight: 700; font-size: 1.3rem; }
    .badge-medium { color: #e17055; font-weight: 700; font-size: 1.3rem; }
    .badge-low    { color: #00b894; font-weight: 700; font-size: 1.3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# data loading
@st.cache_data(show_spinner=False)
def load_all_data():
    try:
        df = load_timeseries()
    except FileNotFoundError:
        with st.spinner("First-time setup: downloading Stack Overflow surveys..."):
            df = build_timeseries()
    return df


@st.cache_data(show_spinner=False)
def run_analysis(_df):
    growth_df = compute_skill_growth(_df)
    survival_df = compute_survival_score(growth_df)
    yoy_df = compute_yoy_growth(_df)
    return growth_df, survival_df, yoy_df


# header
st.markdown('<p class="big-title">AI Job Market Disruption Analyzer</p>', unsafe_allow_html=True)
st.caption("Powered by Stack Overflow Developer Surveys 2019 – 2025")
st.divider()

# load data
with st.spinner("Loading & analysing skill data..."):
    df = load_all_data()
    growth_df, survival_df, yoy_df = run_analysis(df)

# KPI strip
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Unique Skills", f"{df['skill'].nunique():,}")
c2.metric("Years of Data", df["year"].nunique())
c3.metric("Max Survey Size", f"{int(df['total_respondents'].max()):,}")
c4.metric("Growing Skills", len(growth_df[growth_df["slope"] > 0]))
c5.metric("Declining Skills", len(growth_df[growth_df["slope"] < 0]))

st.divider()

# tabs
t1, t2, t3, t4, t5, t6 = st.tabs(
    [
        "Emerging Skills",
        "Declining Skills",
        "AI Disruption Index",
        "Career Recommender",
        "Skill Survival Score",
        "Future Forecast",
    ]
)

# --- tab 1: emerging skills ---
with t1:
    st.subheader("Fastest Growing Skills")
    st.caption("Ranked by linear trend slope (percentage-points gained per year) across SO surveys.")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        top_n = st.slider("Show top N", 10, 40, 20, key="t1_n")

    emerging = get_emerging_skills(growth_df, top_n)

    fig = px.bar(
        emerging,
        x="slope",
        y="skill",
        orientation="h",
        color="slope",
        color_continuous_scale="Viridis",
        labels={"slope": "Growth Rate (pp/year)", "skill": "Skill"},
        title=f"Top {top_n} Fastest Growing Skills (2019 – 2025)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=max(400, top_n * 22))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View data table"):
        st.dataframe(
            emerging.rename(
                columns={
                    "slope": "Growth Rate (pp/yr)",
                    "latest_pct": "Current Demand (%)",
                    "total_growth_pct": "Total Growth (%)",
                    "years_tracked": "Years Tracked",
                    "r_squared": "R²",
                }
            )[["skill","Growth Rate (pp/yr)","Current Demand (%)","Total Growth (%)","Years Tracked","R²"]],
            use_container_width=True,
        )

    st.subheader("Compare Skill Trends Over Time")
    all_skills = sorted(df["skill"].unique().tolist())
    defaults = [s for s in ["Python", "TypeScript", "Rust", "Go"] if s in all_skills]
    selected = st.multiselect("Select skills", all_skills, default=defaults, key="t1_compare")

    if selected:
        cdf = df[df["skill"].isin(selected)]
        fig2 = px.line(
            cdf, x="year", y="pct_respondents",
            color="skill", markers=True,
            labels={"pct_respondents": "% of Respondents", "year": "Year"},
            title="Skill Demand Over Time",
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)


# --- tab 2: declining skills ---
with t2:
    st.subheader("Fastest Declining Skills")
    st.caption("Skills losing developer adoption over time.")

    col_a, col_b = st.columns([3, 1])
    with col_b:
        top_n2 = st.slider("Show top N", 10, 40, 20, key="t2_n")

    declining = get_declining_skills(growth_df, top_n2)

    fig = px.bar(
        declining,
        x="slope",
        y="skill",
        orientation="h",
        color="slope",
        color_continuous_scale="RdYlGn",
        labels={"slope": "Decline Rate (pp/year)", "skill": "Skill"},
        title=f"Top {top_n2} Declining Skills",
    )
    fig.update_layout(yaxis={"categoryorder": "total descending"}, height=max(400, top_n2 * 22))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View data table"):
        st.dataframe(
            declining.rename(
                columns={
                    "slope": "Decline Rate (pp/yr)",
                    "latest_pct": "Current Demand (%)",
                    "total_growth_pct": "Total Change (%)",
                    "years_tracked": "Years Tracked",
                }
            )[["skill","Decline Rate (pp/yr)","Current Demand (%)","Total Change (%)","Years Tracked"]],
            use_container_width=True,
        )

    st.subheader("YoY Growth Rates (Bottom 15)")
    worst_yoy = yoy_df.tail(15).sort_values("avg_yoy_growth_pct")
    fig3 = px.bar(
        worst_yoy, x="avg_yoy_growth_pct", y="skill",
        orientation="h", color="avg_yoy_growth_pct",
        color_continuous_scale="RdYlGn",
        labels={"avg_yoy_growth_pct": "Avg YoY Growth (%)", "skill": "Skill"},
        title="Worst Average Year-over-Year Growth",
    )
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)


# --- tab 3: AI disruption index ---
with t3:
    st.subheader("AI Disruption Index")
    st.caption("Estimate automation risk for any job title using fuzzy matching + keyword heuristics.")

    job_input = st.text_input(
        "Enter a job title",
        placeholder="e.g.  Data Analyst, Software Engineer, Content Writer",
    )

    if job_input.strip():
        result = get_ai_exposure_score(job_input.strip())
        score = result["score"]

        left, right = st.columns([1, 2])

        with left:
            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "AI Disruption Score", "font": {"size": 16}},
                    number={"suffix": " / 100"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "steelblue"},
                        "steps": [
                            {"range": [0, 40], "color": "#d5f5e3"},
                            {"range": [40, 70], "color": "#fef9e7"},
                            {"range": [70, 100], "color": "#fadbd8"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": score,
                        },
                    },
                )
            )
            gauge.update_layout(height=300, margin={"t": 30, "b": 10})
            st.plotly_chart(gauge, use_container_width=True)

        with right:
            st.markdown(f"### {result['job_title'].title()}")
            badge_class = {
                "High Risk": "badge-high",
                "Medium Risk": "badge-medium",
                "Low Risk": "badge-low",
            }[result["risk_level"]]
            st.markdown(
                f'<span class="{badge_class}">{result["risk_level"]}</span>',
                unsafe_allow_html=True,
            )
            st.write("")
            st.write(result["advice"])
            st.caption(f"Match method: {result['method']}")

            if score >= 70:
                st.error("High automation risk detected.")
            elif score >= 40:
                st.warning("Moderate risk — upskilling recommended.")
            else:
                st.success("Relatively safe from AI displacement.")

    with st.expander("Reference: All job roles & exposure scores"):
        ref_df = pd.DataFrame(
            [
                {
                    "Job Role": k.title(),
                    "AI Exposure Score": v,
                    "Risk Level": "High" if v >= 70 else ("Medium" if v >= 40 else "Low"),
                }
                for k, v in sorted(AI_EXPOSURE_SCORES.items(), key=lambda x: -x[1])
            ]
        )
        st.dataframe(ref_df, use_container_width=True, height=400)


# --- tab 4: career recommender ---
with t4:
    st.subheader("Career Recommendation Engine")
    st.caption("Tell us what you know — we'll show you what to learn next.")

    all_skills_list = sorted(df["skill"].unique().tolist())
    defaults_t4 = [s for s in ["Python", "SQL", "JavaScript"] if s in all_skills_list]

    current_skills = st.multiselect(
        "Your current skills",
        all_skills_list,
        default=defaults_t4,
    )
    num_recs = st.slider("Number of recommendations", 3, 10, 5, key="t4_recs")

    if current_skills:
        recs = recommend_skills(current_skills, growth_df, survival_df, num_recs)

        if recs:
            st.subheader("Recommended Next Skills")

            for i, r in enumerate(recs, 1):
                with st.expander(
                    f"#{i}  {r['skill']}  —  Survival Score: {r['survival_score']:.0f} / 100"
                ):
                    ca, cb, cc = st.columns(3)
                    ca.metric("Survival Score", f"{r['survival_score']:.0f}")
                    cb.metric("Current Demand", f"{r['current_demand_pct']:.1f}%")
                    cc.metric("Trend", r["growth_trend"])
                    st.info(r["reason"])

            rec_df = pd.DataFrame(recs)
            fig = px.bar(
                rec_df, x="skill", y="survival_score",
                color="survival_score",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                labels={"survival_score": "Survival Score", "skill": "Skill"},
                title="Recommended Skills — Survival Score",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recommendations found — try removing a few skills from your list.")
    else:
        st.info("Select at least one skill above to get recommendations.")


# --- tab 5: skill survival score ---
with t5:
    st.subheader("Skill Survival Score Leaderboard")
    st.caption(
        "Composite score (0-100):  "
        "Demand Growth (40 pts)  +  AI Resistance (40 pts)  +  Current Demand (20 pts)"
    )

    col_a, col_b = st.columns([3, 1])
    with col_b:
        top_n5 = st.slider("Show top N", 10, 60, 30, key="t5_n")

    top_survival = survival_df.head(top_n5)

    fig_tree = px.treemap(
        top_survival,
        path=["skill"],
        values="survival_score",
        color="survival_score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        title=f"Skill Survival Score — Top {top_n5}",
    )
    fig_tree.update_layout(height=500)
    st.plotly_chart(fig_tree, use_container_width=True)

    # score breakdown
    melted = top_survival.head(20).melt(
        id_vars="skill",
        value_vars=["growth_score", "resistance_score", "demand_score"],
        var_name="Component",
        value_name="Score",
    )
    comp_labels = {
        "growth_score": "Demand Growth",
        "resistance_score": "AI Resistance",
        "demand_score": "Current Demand",
    }
    melted["Component"] = melted["Component"].map(comp_labels)

    fig_stack = px.bar(
        melted, x="skill", y="Score", color="Component",
        barmode="stack",
        color_discrete_map={
            "Demand Growth": "#3498db",
            "AI Resistance": "#2ecc71",
            "Current Demand": "#e67e22",
        },
        title="Score Breakdown — Top 20 Skills",
        labels={"skill": "Skill", "Score": "Score Contribution"},
    )
    fig_stack.update_layout(height=420, xaxis_tickangle=-40)
    st.plotly_chart(fig_stack, use_container_width=True)

    with st.expander("Full leaderboard table"):
        st.dataframe(
            top_survival[[
                "skill", "survival_score",
                "growth_score", "resistance_score", "demand_score",
                "slope", "latest_pct",
            ]].rename(
                columns={
                    "survival_score": "Survival Score",
                    "growth_score": "Growth (0-40)",
                    "resistance_score": "AI Resistance (0-40)",
                    "demand_score": "Current Demand (0-20)",
                    "slope": "Trend Slope",
                    "latest_pct": "Latest Demand (%)",
                }
            ),
            use_container_width=True,
        )


# --- tab 6: future forecast ---
with t6:
    st.subheader("Future Demand Forecast")
    st.caption("Linear extrapolation of SO survey trends to forecast skill demand.")

    col_a, col_b = st.columns(2)
    with col_a:
        forecast_skill = st.selectbox(
            "Select a skill to forecast",
            sorted(df["skill"].unique().tolist()),
            index=sorted(df["skill"].unique().tolist()).index("Python")
            if "Python" in df["skill"].unique()
            else 0,
        )
    with col_b:
        target_year = st.slider("Forecast horizon (year)", 2026, 2030, 2028, key="t6_year")

    if forecast_skill:
        f_result = forecast_demand(df, forecast_skill, target_year)

        if f_result:
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Demand", f"{f_result['current_pct']}%", help="Latest survey year")
            m2.metric(
                f"{target_year} Forecast",
                f"{f_result['predicted_pct']}%",
                delta=f"{f_result['predicted_pct'] - f_result['current_pct']:.1f}pp",
            )
            m3.metric("Trend", f_result["trend"])

            forecast_years = list(range(2026, target_year + 1))
            pred_df = predict_skill_demand(df, forecast_skill, forecast_years)

            if pred_df is not None:
                hist = pred_df[pred_df["row_type"] == "historical"]
                fore = pred_df[pred_df["row_type"] == "forecast"]

                fig_fc = go.Figure()
                fig_fc.add_trace(
                    go.Scatter(
                        x=hist["year"], y=hist["pct_respondents"],
                        mode="lines+markers", name="Historical",
                        line=dict(color="#3498db", width=2.5),
                        marker=dict(size=8),
                    )
                )
                fig_fc.add_trace(
                    go.Scatter(
                        x=pd.concat([hist.tail(1), fore])["year"],
                        y=pd.concat([hist.tail(1), fore])["pct_respondents"],
                        mode="lines+markers", name="Forecast",
                        line=dict(color="#e74c3c", width=2.5, dash="dash"),
                        marker=dict(size=8, symbol="diamond"),
                    )
                )
                fig_fc.update_layout(
                    title=f"{forecast_skill} — Demand Forecast to {target_year}",
                    xaxis_title="Year",
                    yaxis_title="% of SO Respondents",
                    height=420,
                    legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
                )
                st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.warning(f"Not enough data points to forecast '{forecast_skill}'.")

    # batch forecast
    st.subheader(f"Batch Forecast — Top 15 Skills in {target_year}")
    batch_skills = survival_df.head(15)["skill"].tolist()
    batch_records = []

    for s in batch_skills:
        r = forecast_demand(df, s, target_year)
        if r:
            delta = round(r["predicted_pct"] - r["current_pct"], 2)
            batch_records.append(
                {
                    "Skill": r["skill"],
                    "Now (%)": r["current_pct"],
                    f"{target_year} (%)": r["predicted_pct"],
                    "Change (pp)": delta,
                    "Current Label": r["current_demand"],
                    f"{target_year} Label": r["predicted_demand"],
                    "Trend": r["trend"],
                }
            )

    if batch_records:
        batch_df = pd.DataFrame(batch_records)
        st.dataframe(batch_df, use_container_width=True)

        fig_batch = px.bar(
            batch_df, x="Skill", y="Change (pp)",
            color="Change (pp)",
            color_continuous_scale="RdYlGn",
            title=f"Predicted Change in Demand (Now → {target_year})",
            labels={"Change (pp)": "Change (percentage points)"},
        )
        fig_batch.update_layout(height=380, xaxis_tickangle=-35)
        st.plotly_chart(fig_batch, use_container_width=True)


# sidebar
with st.sidebar:
    st.title("Controls")

    if st.button("Refresh Data (re-download)"):
        st.cache_data.clear()
        build_timeseries(force_download=True)
        st.success("Data refreshed!")
        st.rerun()

    st.divider()
    st.subheader("About")
    st.write(
        """
        Analyses 7 years of Stack Overflow Developer Survey data
        (2019-2025) to surface emerging/declining skills,
        job disruption risk, upskilling paths, and demand forecasts.

        **Data:** Stack Overflow Annual Developer Survey
        **Models:** Linear regression, fuzzy matching, composite scoring
        """
    )

    st.divider()
    years_available = sorted(df["year"].unique())
    st.caption(f"Years: {years_available[0]} – {years_available[-1]}")
    st.caption(f"Skills tracked: {df['skill'].nunique():,}")
    st.caption(f"Records: {len(df):,}")
