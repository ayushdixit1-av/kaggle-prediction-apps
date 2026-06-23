import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")

st.markdown(
    """
<style>
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
    @keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
    @keyframes glow { 0%,100% { box-shadow: 0 0 20px rgba(139,92,246,0.2); } 50% { box-shadow: 0 0 50px rgba(139,92,246,0.4); } }
    @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes float { 0%,100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }

    .stAppDeployButton, button[title="Fork this app"], button[title="Deploy this app"] { display: none !important; }
    .main-title {
        background: linear-gradient(90deg, #8b5cf6, #a78bfa, #8b5cf6);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .subtitle { animation: fadeInUp 0.6s ease-out 0.2s both; color: #aaa; }
    .score-card {
        background: rgba(26,26,46,0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid rgba(139,92,246,0.15);
        animation: fadeInUp 0.8s ease-out, glow 3s ease-in-out infinite;
    }
    .score-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffd700;
        animation: fadeInUp 0.8s ease-out;
    }
    .score-label { color: #aaa; font-size: 0.85rem; margin-top: 0.25rem; }
    .score-stars { font-size: 1.5rem; margin-top: 0.25rem; animation: fadeInUp 0.8s ease-out 0.3s both; }
    .metric-box {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px; padding: 0.75rem; text-align: center;
        border: 1px solid rgba(255,255,255,0.05); animation: fadeInUp 0.6s ease-out both;
        transition: all 0.3s ease;
    }
    .metric-box:hover { transform: translateY(-2px); border-color: rgba(139,92,246,0.2); }
    .metric-value { font-size: 1.2rem; font-weight: 700; color: #8b5cf6; }
    .metric-label { font-size: 0.7rem; color: #888; margin-top: 0.15rem; }
    .insight-box {
        background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(139,92,246,0.01));
        border-radius: 12px; padding: 0.75rem 1rem;
        border: 1px solid rgba(139,92,246,0.1);
        animation: fadeInUp 0.8s ease-out 0.4s both;
        font-size: 0.85rem;
    }
    .sidebar-header {
        font-size: 1.2rem; font-weight: 700; color: #8b5cf6;
        margin-bottom: 1rem; animation: pulse 2s ease-in-out infinite;
    }
    .stButton button {
        background: linear-gradient(90deg, #8b5cf6, #7c3aed) !important;
        border: none !important;
        animation: pulse 2s ease-in-out infinite !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem !important;
    }
    .stButton button:hover {
        animation: none !important;
        transform: scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(139,92,246,0.3) !important;
    }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #fff;
        margin: 1rem 0 0.5rem;
        animation: slideIn 0.5s ease-out;
    }
    .wine-icon {
        font-size: 2.5rem;
        animation: float 3s ease-in-out infinite;
        text-align: center;
        margin: 0.25rem 0;
    }
    .input-group {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 0.75rem 0.75rem 0.25rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid rgba(139,92,246,0.3);
    }
    .input-group-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b5cf6;
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

st.markdown('<div class="main-title">🍷 Wine Quality Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict the quality score of a red wine from its chemical composition</div>',
    unsafe_allow_html=True,
)

DATA_PATH = "data/winequality-red.csv"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


@st.cache_data
def load_wine_data():
    try:
        df = pd.read_csv(DATA_PATH, sep=";")
    except FileNotFoundError:
        df = pd.read_csv(DATA_URL, sep=";")
    return df


@st.cache_resource
def train_wine_model(df):
    feature_cols = [c for c in df.columns if c != "quality"]
    X = df[feature_cols]
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return model, scaler, feature_cols


df = load_wine_data()
with st.spinner("🍷 Training model..."):
    model, scaler, feature_cols = train_wine_model(df)

feature_labels = {
    "fixed acidity": "Fixed Acidity", "volatile acidity": "Volatile Acidity",
    "citric acid": "Citric Acid", "residual sugar": "Residual Sugar",
    "chlorides": "Chlorides", "free sulfur dioxide": "Free SO₂",
    "total sulfur dioxide": "Total SO₂", "density": "Density",
    "pH": "pH", "sulphates": "Sulphates", "alcohol": "Alcohol",
}

feature_icons = {
    "fixed acidity": "🧪", "volatile acidity": "🧪", "citric acid": "🍋",
    "residual sugar": "🍬", "chlorides": "🧂", "free sulfur dioxide": "💨",
    "total sulfur dioxide": "💨", "density": "⚖️", "pH": "📏",
    "sulphates": "🧴", "alcohol": "🍾",
}

feature_ranges = {
    "fixed acidity": (4.0, 16.0, 8.5), "volatile acidity": (0.1, 1.6, 0.5),
    "citric acid": (0.0, 1.0, 0.3), "residual sugar": (0.5, 15.0, 2.5),
    "chlorides": (0.01, 0.6, 0.08), "free sulfur dioxide": (1.0, 70.0, 15.0),
    "total sulfur dioxide": (5.0, 290.0, 45.0), "density": (0.9900, 1.0050, 0.9960),
    "pH": (2.8, 4.0, 3.3), "sulphates": (0.3, 2.0, 0.65),
    "alcohol": (8.0, 15.0, 10.5),
}

wine_groups = [
    ("🔬 Acidity", ["fixed acidity", "volatile acidity", "citric acid", "pH"]),
    ("🧪 Composition", ["residual sugar", "chlorides", "density", "sulphates"]),
    ("🍇 Sulfur & Alcohol", ["free sulfur dioxide", "total sulfur dioxide", "alcohol"]),
]

feature_help = {
    "fixed acidity": "Tartaric acid level (g/dm³) — higher = more tart",
    "volatile acidity": "Acetic acid level — high = vinegar taste",
    "citric acid": "Citric acid — adds freshness",
    "residual sugar": "Remaining sugar after fermentation (g/dm³)",
    "chlorides": "Salt content (g/dm³)",
    "free sulfur dioxide": "Free SO₂ — prevents oxidation (mg/dm³)",
    "total sulfur dioxide": "Total SO₂ — bound + free (mg/dm³)",
    "density": "Wine density (g/cm³) — related to sugar/alcohol",
    "pH": "Acidity level — lower = more acidic",
    "sulphates": "SO₂ preservative additive (g/dm³)",
    "alcohol": "Alcohol percentage by volume (%)",
}

if "wine_reset_key" not in st.session_state:
    st.session_state.wine_reset_key = 0

rk = st.session_state.wine_reset_key

inputs = {}
for group_label, group_feats in wine_groups:
    available = [f for f in group_feats if f in feature_cols]
    if not available:
        continue
    st.sidebar.markdown(
        f'<div class="input-group"><div class="input-group-label">{group_label}</div>',
        unsafe_allow_html=True,
    )
    for feat in available:
        min_v, max_v, default = feature_ranges[feat]
        span = max_v - min_v
        if span <= 0.1:
            step = 0.001
        elif span <= 1:
            step = 0.01
        elif span <= 10:
            step = 0.1
        else:
            step = 1.0
        fmt = f"{{:.{max(0, -int(round(np.log10(step))))}f}}"
        icon = feature_icons.get(feat, "•")
        label = feature_labels.get(feat, feat)
        inputs[feat] = st.sidebar.slider(
            f"{icon} {label}", min_value=float(min_v), max_value=float(max_v),
            value=float(default), step=float(step), format=fmt,
            help=feature_help.get(feat, ""), key=f"w_{feat[:3]}_{rk}",
        )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")

col_a, col_b = st.sidebar.columns(2)
predicted = False
with col_a:
    if st.button("🍷 Predict", type="primary", use_container_width=True):
        predicted = True
with col_b:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.wine_reset_key += 1
        st.rerun()

st.sidebar.markdown(
    '<div class="sidebar-info">Adjust chemistry values and click Predict</div>',
    unsafe_allow_html=True,
)

if predicted:
    input_df = pd.DataFrame([inputs])[feature_cols]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    pred_clipped = np.clip(pred, 0, 10)

    stars = "⭐" * int(round(pred_clipped))
    if pred_clipped >= 7:
        quality_label = "🥇 Excellent"
        quality_color = "#ffd700"
    elif pred_clipped >= 5:
        quality_label = "👍 Good"
        quality_color = "#22c55e"
    else:
        quality_label = "👎 Below Average"
        quality_color = "#ef4444"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="wine-icon">🍷</div>
                <div class="score-number">{pred_clipped:.1f}</div>
                <div class="score-label">Predicted Quality Score</div>
                <div class="score-stars">{stars}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-box" style="animation-delay:0.2s">
                <div class="metric-value" style="color:{quality_color};">{quality_label}</div>
                <div class="metric-label">Rating</div>
            </div>
            <div class="metric-box" style="animation-delay:0.4s;margin-top:1rem;">
                <div class="metric-value">{model.score(scaler.transform(df[feature_cols]), df["quality"]):.2f}</div>
                <div class="metric-label">Model R² Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        avg_quality = df["quality"].mean()
        diff = pred_clipped - avg_quality
        st.markdown(
            f"""
            <div class="metric-box" style="animation-delay:0.3s">
                <div class="metric-value" style="color:{'#22c55e' if diff >= 0 else '#ef4444'};">{diff:+.1f}</div>
                <div class="metric-label">vs Dataset Average ({avg_quality:.1f})</div>
            </div>
            <div class="metric-box" style="animation-delay:0.5s;margin-top:1rem;">
                <div class="metric-value">Random Forest</div>
                <div class="metric-label">Model (200 trees)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">📊 Feature Importance</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        "Feature": [feature_labels.get(f, f) for f in feature_cols],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)
    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Purples",
    )
    fig.update_layout(
        height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📈 Your Wine vs Dataset</div>', unsafe_allow_html=True)
    fig2 = px.histogram(
        df, x="quality", nbins=10,
        color_discrete_sequence=["rgba(139,92,246,0.3)"],
    )
    fig2.add_vline(
        x=pred_clipped, line_dash="dash", line_color="#ffd700",
        annotation_text="Your Wine", annotation_font_color="#ffd700",
    )
    fig2.update_layout(
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    if pred_clipped >= 7:
        st.markdown(
            '<div class="insight-box">🥇 <strong>Excellent wine!</strong> This would rank among the top '
            'wines in the dataset. High alcohol and low volatile acidity are key indicators of quality.</div>',
            unsafe_allow_html=True,
        )
        st.balloons()
    elif pred_clipped >= 5:
        st.markdown(
            '<div class="insight-box">👍 <strong>Good quality wine.</strong> Try increasing alcohol content '
            'or reducing volatile acidity to push it into the excellent range.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="insight-box">📉 <strong>Below average.</strong> Consider adjusting: higher alcohol '
            '(>11%), lower volatile acidity (<0.5), and higher sulphates typically improve quality.</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("👈 Adjust the wine chemistry sliders in the sidebar and click **Predict Quality**.")
    st.markdown(
        f"<div style='background:rgba(139,92,246,0.05);border-radius:16px;padding:1.25rem;margin-top:1rem;'>"
        f"<span style='color:#8b5cf6;font-weight:700;'>{len(df):,}</span> "
        f"<span style='color:#888;'>wines in dataset · Avg quality: "
        f"<span style='color:#ffd700;'>{df['quality'].mean():.1f}/10</span></span></div>",
        unsafe_allow_html=True,
    )
