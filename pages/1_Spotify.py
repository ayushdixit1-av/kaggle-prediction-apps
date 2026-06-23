import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Spotify Predictor", page_icon="🎵", layout="wide")

st.markdown(
    """
<style>
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
    @keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
    @keyframes countUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes glow { 0%,100% { box-shadow: 0 0 20px rgba(29,185,84,0.2); } 50% { box-shadow: 0 0 40px rgba(29,185,84,0.4); } }
    @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }

    .stAppDeployButton, button[title="Fork this app"], button[title="Deploy this app"] { display: none !important; }
    .main-title {
        background: linear-gradient(90deg, #1DB954, #1ed760, #1DB954);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .subtitle { animation: fadeInUp 0.6s ease-out 0.2s both; color: #aaa; }
    .result-card {
        background: rgba(13,43,26,0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.25rem;
        border: 1px solid rgba(29,185,84,0.15);
        animation: fadeInUp 0.8s ease-out, glow 3s ease-in-out infinite;
        text-align: center;
    }
    .result-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1DB954;
        animation: countUp 1s ease-out;
    }
    .result-label { color: #aaa; font-size: 0.85rem; margin-top: 0.25rem; }
    .metric-box {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 0.75rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.05);
        animation: fadeInUp 0.6s ease-out both;
        transition: all 0.3s ease;
    }
    .metric-box:hover { transform: translateY(-2px); border-color: rgba(29,185,84,0.2); }
    .metric-value { font-size: 1.2rem; font-weight: 700; color: #1DB954; }
    .metric-label { font-size: 0.7rem; color: #888; margin-top: 0.15rem; }
    .insight-box {
        background: linear-gradient(135deg, rgba(29,185,84,0.08), rgba(29,185,84,0.01));
        border-radius: 12px;
        padding: 0.75rem 1rem;
        border: 1px solid rgba(29,185,84,0.1);
        animation: fadeInUp 0.8s ease-out 0.4s both;
        font-size: 0.85rem;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1DB954;
        margin-bottom: 1rem;
        animation: pulse 2s ease-in-out infinite;
    }
    .stButton button {
        background: linear-gradient(90deg, #1DB954, #169c46) !important;
        border: none !important;
        animation: pulse 2s ease-in-out infinite !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        animation: none !important;
        transform: scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(29,185,84,0.3) !important;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #fff;
        margin: 1rem 0 0.5rem;
        animation: slideIn 0.5s ease-out;
    }
    .stMetric { animation: fadeInUp 0.6s ease-out both; }
    .input-group {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 0.75rem 0.75rem 0.25rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid rgba(29,185,84,0.3);
    }
    .input-group-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #1DB954;
        margin-bottom: 0.5rem;
        padding-left: 0.25rem;
    }
    .sidebar-info {
        font-size: 0.75rem;
        color: #666;
        text-align: center;
        padding: 0.5rem;
        margin-top: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .stSidebar .stSlider { padding-bottom: 0.25rem; }
    .stSidebar label { font-size: 0.85rem !important; font-weight: 500 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🎵 Spotify Stream Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict how many streams a song would get based on its audio characteristics</div>',
    unsafe_allow_html=True,
)

DATA_PATH = "data/spotify.csv"
DATA_URL = "https://raw.githubusercontent.com/shiran1shaked1shalev1/Top-Spotify-Songs-2023/main/spotify-2023.csv"


@st.cache_data
def load_spotify_data():
    try:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")
    except FileNotFoundError:
        df = pd.read_csv(DATA_URL, encoding="latin-1")
    df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
    df = df.dropna(subset=["streams"])
    feat_cols = {
        "bpm": "bpm",
        "danceability_%": "danceability",
        "valence_%": "valence",
        "energy_%": "energy",
        "acousticness_%": "acousticness",
        "speechiness_%": "speechiness",
        "liveness_%": "liveness",
    }
    rename = {k: v for k, v in feat_cols.items() if k in df.columns}
    df = df.rename(columns=rename)
    avail = [v for v in feat_cols.values() if v in df.columns]
    return df, avail


@st.cache_resource
def train_spotify_model(df, features):
    X = df[features].fillna(df[features].median())
    y = np.log1p(df["streams"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X.columns.tolist()


df, features = load_spotify_data()
with st.spinner("🎧 Training model..."):
    model, scaler, feat_names = train_spotify_model(df, features)

slider_configs = {
    "bpm": (60, 200, 120),
    "danceability": (0, 100, 65),
    "valence": (0, 100, 50),
    "energy": (0, 100, 60),
    "acousticness": (0, 100, 20),
    "speechiness": (0, 100, 10),
    "liveness": (0, 100, 15),
}

feature_help = {
    "bpm": "Tempo of the song in beats per minute",
    "danceability": "How suitable the track is for dancing (0-100)",
    "valence": "Musical positiveness conveyed by the track (0-100)",
    "energy": "Perceptual measure of intensity and activity (0-100)",
    "acousticness": "Confidence that the track is acoustic (0-100)",
    "speechiness": "Presence of spoken words in the track (0-100)",
    "liveness": "Likelihood the track was performed live (0-100)",
}

feature_icons = {
    "bpm": "🎵", "danceability": "💃", "valence": "😊",
    "energy": "⚡", "acousticness": "🎸", "speechiness": "🎤", "liveness": "🎭",
}

groups = [
    ("🎵 Rhythm", ["bpm", "danceability"]),
    ("⚡ Energy", ["energy", "acousticness", "liveness"]),
    ("🎭 Mood", ["valence", "speechiness"]),
]

defaults = {k: v[2] for k, v in slider_configs.items() if k in feat_names}
if "reset" not in st.session_state:
    st.session_state.reset = False

inputs = {}
for group_label, group_feats in groups:
    available = [f for f in group_feats if f in feat_names]
    if not available:
        continue
    st.sidebar.markdown(
        f'<div class="input-group"><div class="input-group-label">{group_label}</div>',
        unsafe_allow_html=True,
    )
    for feat in available:
        min_v, max_v, default = slider_configs[feat]
        icon = feature_icons.get(feat, "•")
        label = f"{icon} {feat.capitalize()}"
        if st.session_state.reset:
            default = defaults[feat]
        inputs[feat] = st.sidebar.slider(
            label, min_value=min_v, max_value=max_v, value=default,
            help=feature_help.get(feat, ""), key=f"sp_{feat}",
        )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")

col_a, col_b = st.sidebar.columns(2)
predicted = False
with col_a:
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        predicted = True
with col_b:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.reset = True
        st.rerun()

if st.session_state.reset:
    st.session_state.reset = False

st.sidebar.markdown(
    '<div class="sidebar-info">Adjust features and click Predict</div>',
    unsafe_allow_html=True,
)

if predicted:
    input_df = pd.DataFrame([inputs])[feat_names]
    input_scaled = scaler.transform(input_df)
    pred_log = model.predict(input_scaled)[0]
    pred_streams = int(np.expm1(pred_log))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-number">{pred_streams:,}</div>
                <div class="result-label">Predicted Streams</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        r2 = model.score(
            scaler.transform(df[feat_names].fillna(df[feat_names].median())),
            np.log1p(df["streams"]),
        )
        st.markdown(
            f"""
            <div class="metric-box" style="animation-delay:0.2s">
                <div class="metric-value">{r2:.2f}</div>
                <div class="metric-label">Model R² Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        percentile = int((df["streams"] < pred_streams).mean() * 100)
        st.markdown(
            f"""
            <div class="metric-box" style="animation-delay:0.4s">
                <div class="metric-value">Top {100 - percentile}%</div>
                <div class="metric-label">Percentile Rank</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">📊 Feature Importance</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": model.feature_importances_})
    imp_df = imp_df.sort_values("Importance", ascending=True)
    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Greens",
        title="",
    )
    fig.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📈 Your Song vs Dataset</div>', unsafe_allow_html=True)
    fig2 = px.histogram(
        df, x="streams", nbins=50,
        color_discrete_sequence=["rgba(29,185,84,0.3)"],
    )
    fig2.add_vline(
        x=pred_streams, line_dash="dash", line_color="#1DB954",
        annotation_text="Your Song", annotation_font_color="#1DB954",
    )
    fig2.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    if pred_streams > df["streams"].median():
        st.markdown(
            f'<div class="insight-box">🎉 <strong>Hit potential!</strong> This song would outperform '
            f'{percentile}% of top Spotify songs. Consider increasing danceability or energy for even more streams.</div>',
            unsafe_allow_html=True,
        )
        st.balloons()
    else:
        st.markdown(
            f'<div class="insight-box">📊 <strong>Room for improvement.</strong> Try increasing bpm, '
            f'danceability, or energy — these tend to boost stream counts.</div>',
            unsafe_allow_html=True,
        )
else:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("👈 Adjust the audio features in the sidebar and click **Predict Streams** to see the magic!")
    with col2:
        st.markdown(
            f"<div style='background:rgba(29,185,84,0.05);border-radius:16px;padding:1rem;text-align:center;"
            f"animation:pulse 3s ease-in-out infinite;'>"
            f"<div style='font-size:2rem;color:#1DB954;font-weight:700;'>{len(df):,}</div>"
            f"<div style='color:#888;font-size:0.85rem;'>Songs in Dataset</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
