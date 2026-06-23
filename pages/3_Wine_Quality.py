import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="Wine Quality Predictor", page_icon="🍷", layout="wide")
st.title("🍷 Wine Quality Predictor")
st.markdown("Predict the quality score of a red wine (0-10) based on its chemical composition.")

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
with st.spinner("Training model..."):
    model, scaler, feature_cols = train_wine_model(df)

feature_labels = {
    "fixed acidity": "Fixed Acidity",
    "volatile acidity": "Volatile Acidity",
    "citric acid": "Citric Acid",
    "residual sugar": "Residual Sugar",
    "chlorides": "Chlorides",
    "free sulfur dioxide": "Free Sulfur Dioxide",
    "total sulfur dioxide": "Total Sulfur Dioxide",
    "density": "Density",
    "pH": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol",
}

feature_ranges = {
    "fixed acidity": (4.0, 16.0, 8.5),
    "volatile acidity": (0.1, 1.6, 0.5),
    "citric acid": (0.0, 1.0, 0.3),
    "residual sugar": (0.5, 15.0, 2.5),
    "chlorides": (0.01, 0.6, 0.08),
    "free sulfur dioxide": (1, 70, 15),
    "total sulfur dioxide": (5, 290, 45),
    "density": (0.990, 1.005, 0.997),
    "pH": (2.8, 4.0, 3.3),
    "sulphates": (0.3, 2.0, 0.65),
    "alcohol": (8.0, 15.0, 10.5),
}

st.sidebar.header("🧪 Wine Chemistry")

inputs = {}
for feat in feature_cols:
    label = feature_labels.get(feat, feat)
    if feat in feature_ranges:
        min_v, max_v, default = feature_ranges[feat]
        step = 0.01 if max_v - min_v < 10 else 0.1
        inputs[feat] = st.sidebar.slider(label, min_value=min_v, max_value=max_v, value=default, step=step, format="%.2f")
    else:
        inputs[feat] = st.sidebar.number_input(label, value=float(df[feat].median()))

if st.sidebar.button("🍷 Predict Quality", type="primary", use_container_width=True):
    input_df = pd.DataFrame([inputs])[feature_cols]
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    pred_clipped = np.clip(pred, 0, 10)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div style="text-align:center;padding:2rem;border-radius:1rem;background-color:#1b1b3a">
                <h1 style="color:#ffd700;font-size:4rem;margin:0">{pred_clipped:.1f}</h1>
                <p style="color:#aaa;font-size:1.2rem">Predicted Quality Score</p>
                <p style="color:#666">(out of 10)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        quality_label = (
            "🍷 Excellent" if pred_clipped >= 7 else
            "👍 Good" if pred_clipped >= 5 else
            "👎 Below Average"
        )
        st.metric("Rating", quality_label)
        st.metric("Model", "Random Forest (200 trees)")
    with col3:
        r2 = model.score(scaler.transform(df[feature_cols]), df["quality"])
        st.metric("Model R²", f"{r2:.2f}")
        avg_quality = df["quality"].mean()
        st.metric("vs Dataset Avg", f"{pred_clipped - avg_quality:+.1f}")

    st.markdown("### 📊 Feature Importance")
    imp_df = pd.DataFrame({
        "Feature": [feature_labels.get(f, f) for f in feature_cols],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Purples",
        title="Which chemical features determine wine quality?",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Your Wine vs Dataset Distribution")
    fig2 = px.histogram(
        df,
        x="quality",
        nbins=10,
        title="Quality Score Distribution (training data)",
        color_discrete_sequence=["#8B4513"],
    )
    fig2.add_vline(x=pred_clipped, line_dash="dash", line_color="gold", annotation_text="Your wine")
    st.plotly_chart(fig2, use_container_width=True)

    if pred_clipped >= 7:
        st.success("🥇 Excellent quality! This wine would rank among the best in the dataset.")
    elif pred_clipped >= 5:
        st.info("✅ Good quality wine, above or near average.")
    else:
        st.warning("📉 Below average quality. Try adjusting alcohol content or acidity.")
else:
    st.info("👈 Adjust the wine chemistry sliders in the sidebar and click **Predict Quality**.")

    with st.expander("📋 About the Dataset"):
        st.markdown(
            f"""
            - **Samples:** {len(df):,} red wines
            - **Features:** {len(feature_cols)} chemical properties
            - **Quality range:** {int(df['quality'].min())} – {int(df['quality'].max())}
            - **Average quality:** {df['quality'].mean():.1f}
            - **Source:** UCI Machine Learning Repository
            """
        )
