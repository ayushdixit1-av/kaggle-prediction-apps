import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Spotify Predictor", page_icon="🎵", layout="wide")
st.title("🎵 Spotify Stream Predictor")
st.markdown("Predict how many streams a song would get based on its audio characteristics.")

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X.columns.tolist()


df, features = load_spotify_data()
with st.spinner("Training model..."):
    model, scaler, feat_names = train_spotify_model(df, features)

st.sidebar.header("🎛️ Adjust Song Features")

slider_configs = {
    "bpm": (60, 200, 120),
    "danceability": (0, 100, 65),
    "valence": (0, 100, 50),
    "energy": (0, 100, 60),
    "acousticness": (0, 100, 20),
    "speechiness": (0, 100, 10),
    "liveness": (0, 100, 15),
}

inputs = {}
for feat, (min_v, max_v, default) in slider_configs.items():
    if feat in feat_names:
        inputs[feat] = st.sidebar.slider(
            feat.capitalize(),
            min_value=min_v,
            max_value=max_v,
            value=default,
            help=f"{feat.capitalize()} of the song (0-100 or bpm)",
        )

if st.sidebar.button("🚀 Predict Streams", type="primary", use_container_width=True):
    input_df = pd.DataFrame([inputs])[feat_names]
    input_scaled = scaler.transform(input_df)
    pred_log = model.predict(input_scaled)[0]
    pred_streams = int(np.expm1(pred_log))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted Streams", f"{pred_streams:,}", delta=None)
    with col2:
        st.metric("Confidence (R²)", f"{model.score(scaler.transform(df[feat_names].fillna(df[feat_names].median())), np.log1p(df['streams'])):.2f}")
    with col3:
        st.metric("Model Type", "Random Forest (200 trees)")

    st.markdown("### 📊 Feature Importance")
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": model.feature_importances_})
    imp_df = imp_df.sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Greens",
        title="Which features matter most for predicting streams?",
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Stream Distribution in Training Data")
    fig2 = px.histogram(
        df,
        x="streams",
        nbins=50,
        title="Distribution of Streams (training data)",
        color_discrete_sequence=["#1DB954"],
    )
    fig2.add_vline(x=pred_streams, line_dash="dash", line_color="red", annotation_text="Your song")
    st.plotly_chart(fig2, use_container_width=True)

    if pred_streams > df["streams"].median():
        st.success(f"🎉 This song would outperform {int((df['streams'] < pred_streams).mean() * 100)}% of top Spotify songs!")
    else:
        st.info(f"📊 This is in the lower {int((df['streams'] < pred_streams).mean() * 100)}% of top songs — try higher danceability or energy!")

else:
    st.info("👈 Adjust the sliders in the sidebar and click **Predict Streams** to get started!")

    with st.expander("📋 About the Dataset"):
        st.markdown(
            f"""
            - **Rows:** {len(df):,} songs
            - **Features:** {len(features)} audio characteristics
            - **Target:** Stream count
            - **Median streams:** {int(df['streams'].median()):,}
            - **Top song:** {df.loc[df['streams'].idxmax(), 'track_name'] if 'track_name' in df.columns else 'N/A'}
            """
        )
