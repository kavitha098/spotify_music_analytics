import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Spotify Music Analytics Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background: #161B22;
    padding: 15px;
    border-radius: 12px;
    text-align:center;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
}

.metric-value {
    font-size: 28px;
    font-weight: bold;
    color: #1DB954;
}

.metric-label {
    font-size: 14px;
    color: #AAAAAA;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# =====================================================
# HEADER
# =====================================================

st.title("🎵 Spotify Music Analytics Dashboard")
st.markdown("Deep Exploratory Data Analysis for Spotify Songs Dataset")

st.markdown("---")

# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.title("🎛 Filters")

liked_filter = st.sidebar.multiselect(
    "Liked Status",
    options=sorted(df["liked"].unique()),
    default=sorted(df["liked"].unique())
)

df_filtered = df[df["liked"].isin(liked_filter)]

# =====================================================
# KPI SECTION
# =====================================================

total_songs = len(df_filtered)

liked_songs = int(df_filtered["liked"].sum())

avg_energy = round(df_filtered["energy"].mean(), 2)

avg_dance = round(df_filtered["danceability"].mean(), 2)

avg_tempo = round(df_filtered["tempo"].mean(), 1)

avg_duration = round(
    df_filtered["duration_ms"].mean() / 60000,
    2
)

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Songs", total_songs)
c2.metric("Liked Songs", liked_songs)
c3.metric("Avg Energy", avg_energy)
c4.metric("Danceability", avg_dance)
c5.metric("Tempo", avg_tempo)
c6.metric("Duration (Min)", avg_duration)

st.markdown("---")

# =====================================================
# FEATURE DISTRIBUTION
# =====================================================

st.subheader("📊 Feature Distribution Analysis")

features = [
    "danceability",
    "energy",
    "acousticness",
    "speechiness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "loudness"
]

selected_feature = st.selectbox(
    "Select Feature",
    features
)

fig = px.histogram(
    df_filtered,
    x=selected_feature,
    nbins=30,
    marginal="box",
    title=f"{selected_feature} Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# LIKED VS DISLIKED
# =====================================================

st.subheader("❤️ Liked vs Disliked Comparison")

comparison_feature = st.selectbox(
    "Choose Feature",
    features,
    key="compare"
)

fig = px.box(
    df,
    x="liked",
    y=comparison_feature,
    color="liked",
    title=f"Liked vs Disliked ({comparison_feature})"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# RADAR CHART
# =====================================================

st.subheader("🎯 Average Audio Fingerprint")

radar_features = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "speechiness",
    "liveness"
]

radar_values = (
    df_filtered[radar_features]
    .mean()
    .values
)

fig = go.Figure()

fig.add_trace(
    go.Scatterpolar(
        r=radar_values,
        theta=radar_features,
        fill="toself",
        name="Average Profile"
    )
)

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=False,
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# SCATTER ANALYSIS
# =====================================================

st.subheader("📈 Audio Relationship Explorer")

x_axis = st.selectbox(
    "X Axis",
    features,
    index=0
)

y_axis = st.selectbox(
    "Y Axis",
    features,
    index=1
)

fig = px.scatter(
    df_filtered,
    x=x_axis,
    y=y_axis,
    color="liked",
    size="tempo",
    hover_data=["duration_ms"],
    title=f"{x_axis} vs {y_axis}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# TEMPO ANALYSIS
# =====================================================

st.subheader("🥁 Tempo Analysis")

fig = px.histogram(
    df_filtered,
    x="tempo",
    color="liked",
    nbins=25,
    title="Tempo Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# DURATION ANALYSIS
# =====================================================

st.subheader("⏱ Song Duration Analysis")

fig = px.box(
    df_filtered,
    y="duration_ms",
    color="liked",
    title="Song Duration Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# CORRELATION HEATMAP
# =====================================================

st.subheader("🔥 Correlation Matrix")

numeric_df = df_filtered.select_dtypes(
    include=np.number
)

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# TOP CORRELATIONS
# =====================================================

st.subheader("📌 Strongest Correlations")

corr_matrix = corr.abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

top_corr = (
    upper.stack()
    .sort_values(ascending=False)
    .head(10)
)

top_corr_df = pd.DataFrame(
    top_corr
).reset_index()

top_corr_df.columns = [
    "Feature 1",
    "Feature 2",
    "Correlation"
]

st.dataframe(
    top_corr_df,
    use_container_width=True
)

# =====================================================
# FEATURE IMPORTANCE TO LIKED
# =====================================================

st.subheader("🎵 Relationship with Liked Songs")

liked_corr = (
    corr["liked"]
    .sort_values(ascending=False)
)

fig = px.bar(
    liked_corr,
    title="Correlation with Liked Songs"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================================
# AUTOMATED INSIGHTS
# =====================================================

st.subheader("🧠 Automated Insights")

liked_df = df[df["liked"] == 1]
disliked_df = df[df["liked"] == 0]

if len(liked_df) > 0 and len(disliked_df) > 0:

    energy_diff = (
        liked_df["energy"].mean()
        - disliked_df["energy"].mean()
    ) * 100

    dance_diff = (
        liked_df["danceability"].mean()
        - disliked_df["danceability"].mean()
    ) * 100

    valence_diff = (
        liked_df["valence"].mean()
        - disliked_df["valence"].mean()
    ) * 100

    st.success(f"""
### Key Findings

✅ Liked songs have **{energy_diff:.2f}% higher energy**

✅ Liked songs have **{dance_diff:.2f}% higher danceability**

✅ Liked songs have **{valence_diff:.2f}% higher positivity (valence)**

✅ Average tempo of liked songs:
**{liked_df['tempo'].mean():.1f} BPM**

✅ Average duration of liked songs:
**{liked_df['duration_ms'].mean()/60000:.2f} minutes**
""")

# =====================================================
# DATA EXPLORER
# =====================================================

st.subheader("📂 Dataset Explorer")

st.dataframe(
    df_filtered,
    use_container_width=True
)

# =====================================================
# DOWNLOAD DATA
# =====================================================

csv = df_filtered.to_csv(index=False)

st.download_button(
    label="⬇ Download Filtered Data",
    data=csv,
    file_name="filtered_spotify_data.csv",
    mime="text/csv"
)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption(
    "Built with Streamlit • Plotly • Pandas"
)
