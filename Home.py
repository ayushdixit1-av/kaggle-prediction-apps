import streamlit as st

st.set_page_config(
    page_title="Kaggle Predictor Hub",
    page_icon="🤖",
    layout="wide",
)

st.title("🎯 3-in-1 ML Predictor Playground")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎵 Spotify Hit Predictor")
    st.markdown(
        """
        Predict how many streams a song would get based on its audio features.
        
        **Features:** bpm, danceability, energy, valence, acousticness, speechiness, liveness
        
        *Dataset: Top Spotify Songs 2023*
        """
    )
    st.page_link("pages/1_Spotify.py", label="→ Open Spotify Predictor", use_container_width=True)

with col2:
    st.markdown("### 👥 HR Attrition Predictor")
    st.markdown(
        """
        Predict if an employee is likely to leave the company.
        
        **Features:** age, salary, job role, overtime, satisfaction, work-life balance
        
        *Dataset: IBM HR Analytics*
        """
    )
    st.page_link("pages/2_HR_Attrition.py", label="→ Open HR Predictor", use_container_width=True)

with col3:
    st.markdown("### 🍷 Wine Quality Predictor")
    st.markdown(
        """
        Predict the quality score of a red wine from its chemical properties.
        
        **Features:** acidity, sugar, alcohol, pH, sulfates, density
        
        *Dataset: UCI Wine Quality*
        """
    )
    st.page_link("pages/3_Wine_Quality.py", label="→ Open Wine Predictor", use_container_width=True)

st.markdown("---")
st.markdown(
    "Built with ❤️ using Streamlit & scikit-learn | "
    "[Kaggle](https://www.kaggle.com/) | "
    "GitHub repo in comments 👇"
)
