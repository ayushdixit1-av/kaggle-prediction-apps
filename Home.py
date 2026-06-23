import streamlit as st

st.set_page_config(
    page_title="Kaggle Predictor Hub",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
<style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }
    @keyframes blink {
        50% { border-color: transparent; }
    }

    .gradient-text {
        background: linear-gradient(90deg, #1DB954, #1ed760, #1DB954, #169c46);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite;
        font-size: 3rem;
        font-weight: 800;
        display: inline-block;
    }
    .subtitle {
        animation: fadeInUp 0.8s ease-out 0.2s both;
        color: #aaa;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.05);
        animation: fadeInUp 0.8s ease-out both;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(29,185,84,0.03) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.6s ease;
    }
    .card:hover::before {
        opacity: 1;
    }
    .card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(29,185,84,0.3);
        box-shadow: 0 20px 60px rgba(29,185,84,0.15);
    }
    .card-spotify { animation-delay: 0.1s; }
    .card-hr { animation-delay: 0.3s; }
    .card-wine { animation-delay: 0.5s; }
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    .card-spotify .card-icon { animation-delay: 0s; }
    .card-hr .card-icon { animation-delay: 1s; }
    .card-wine .card-icon { animation-delay: 2s; }
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #fff;
    }
    .card-desc {
        color: #b0b0b0;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }
    .card-features {
        color: #666;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }
    .stPageLink {
        animation: pulse 2s ease-in-out infinite !important;
        transition: all 0.3s ease !important;
    }
    .stPageLink:hover {
        animation: none !important;
        transform: scale(1.05) !important;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        animation: fadeIn 1s ease-out 1s both;
        color: #666;
    }
    .tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .tag-green { background: rgba(29,185,84,0.15); color: #1DB954; }
    .tag-blue { background: rgba(59,130,246,0.15); color: #3b82f6; }
    .tag-purple { background: rgba(139,92,246,0.15); color: #8b5cf6; }
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.75rem 1.25rem;
        text-align: center;
        flex: 1;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1DB954;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    hr {
        border-color: rgba(255,255,255,0.05) !important;
    }
    .glow {
        box-shadow: 0 0 40px rgba(29,185,84,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="gradient-text">🎯 3-in-1 ML Predictor Hub</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">🚀 Three interactive machine learning apps. Pick one and start predicting.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown(
        f"""
        <div class="card card-spotify glow">
            <div class="card-icon">🎵</div>
            <div class="card-title">Spotify Stream Predictor</div>
            <div class="card-desc">Predict how many streams a song would get based on its audio characteristics.</div>
            <div class="stats-row">
                <div class="stat"><div class="stat-value">7</div><div class="stat-label">Features</div></div>
                <div class="stat"><div class="stat-value">950+</div><div class="stat-label">Songs</div></div>
                <div class="stat"><div class="stat-value">🎧</div><div class="stat-label">Audio</div></div>
            </div>
            <div class="card-features">
                <span class="tag tag-green">bpm</span>
                <span class="tag tag-green">danceability</span>
                <span class="tag tag-green">energy</span>
                <span class="tag tag-green">valence</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Spotify.py", label="🎵 Open Spotify Predictor →", use_container_width=True)

with col2:
    st.markdown(
        f"""
        <div class="card card-hr glow">
            <div class="card-icon">👥</div>
            <div class="card-title">HR Attrition Predictor</div>
            <div class="card-desc">Predict if an employee is likely to leave the company based on their profile.</div>
            <div class="stats-row">
                <div class="stat"><div class="stat-value">16+</div><div class="stat-label">Features</div></div>
                <div class="stat"><div class="stat-value">1.4K</div><div class="stat-label">Employees</div></div>
                <div class="stat"><div class="stat-value">🏢</div><div class="stat-label">HR</div></div>
            </div>
            <div class="card-features">
                <span class="tag tag-blue">age</span>
                <span class="tag tag-blue">salary</span>
                <span class="tag tag-blue">overtime</span>
                <span class="tag tag-blue">satisfaction</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_HR_Attrition.py", label="👥 Open HR Predictor →", use_container_width=True)

with col3:
    st.markdown(
        f"""
        <div class="card card-wine glow">
            <div class="card-icon">🍷</div>
            <div class="card-title">Wine Quality Predictor</div>
            <div class="card-desc">Predict the quality score of a red wine from its chemical composition.</div>
            <div class="stats-row">
                <div class="stat"><div class="stat-value">11</div><div class="stat-label">Features</div></div>
                <div class="stat"><div class="stat-value">1.6K</div><div class="stat-label">Wines</div></div>
                <div class="stat"><div class="stat-value">🧪</div><div class="stat-label">Chemistry</div></div>
            </div>
            <div class="card-features">
                <span class="tag tag-purple">alcohol</span>
                <span class="tag tag-purple">pH</span>
                <span class="tag tag-purple">sugar</span>
                <span class="tag tag-purple">acidity</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Wine_Quality.py", label="🍷 Open Wine Predictor →", use_container_width=True)

st.markdown("---")
st.markdown(
    '<div class="footer">Built with ❤️ using Streamlit & scikit-learn · '
    '<a href="https://www.kaggle.com/" style="color:#1DB954;">Kaggle</a> · '
    '<a href="https://github.com/ayushdixit1-av/kaggle-prediction-apps" style="color:#1DB954;">GitHub</a></div>',
    unsafe_allow_html=True,
)
