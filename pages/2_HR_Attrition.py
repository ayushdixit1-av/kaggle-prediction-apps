import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")

st.markdown(
    """
<style>
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
    @keyframes shimmer { 0% { background-position: -200% center; } 100% { background-position: 200% center; } }
    @keyframes glow { 0%,100% { box-shadow: 0 0 20px rgba(59,130,246,0.2); } 50% { box-shadow: 0 0 50px rgba(59,130,246,0.4); } }
    @keyframes slideIn { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }
    @keyframes progressFill { from { width: 0%; } to { width: var(--target); } }

    .stAppDeployButton, button[title="Fork this app"], button[title="Deploy this app"] { display: none !important; }
    .main-title {
        background: linear-gradient(90deg, #3b82f6, #60a5fa, #3b82f6);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 3s linear infinite;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .subtitle { animation: fadeInUp 0.6s ease-out 0.2s both; color: #aaa; }
    .risk-card {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        animation: fadeInUp 0.8s ease-out, glow 3s ease-in-out infinite;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .risk-low { background: rgba(13,43,26,0.5); border: 1px solid rgba(34,197,94,0.2); }
    .risk-medium { background: rgba(43,42,26,0.5); border: 1px solid rgba(234,179,8,0.2); }
    .risk-high { background: rgba(43,26,26,0.5); border: 1px solid rgba(239,68,68,0.2); }
    .risk-percent { font-size: 3rem; font-weight: 800; }
    .risk-label { font-size: 0.9rem; margin-top: 0.25rem; }
    .risk-badge { display: inline-block; padding: 0.3rem 1rem; border-radius: 30px; font-weight: 700; margin-top: 0.5rem; font-size: 0.85rem; }
    .metric-box {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px; padding: 0.75rem; text-align: center;
        border: 1px solid rgba(255,255,255,0.05); animation: fadeInUp 0.6s ease-out both;
        transition: all 0.3s ease;
    }
    .metric-box:hover { transform: translateY(-2px); border-color: rgba(59,130,246,0.2); }
    .metric-value { font-size: 1.2rem; font-weight: 700; color: #3b82f6; }
    .metric-label { font-size: 0.7rem; color: #888; margin-top: 0.15rem; }
    .insight-box {
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(59,130,246,0.01));
        border-radius: 12px; padding: 0.75rem 1rem;
        border: 1px solid rgba(59,130,246,0.1);
        animation: fadeInUp 0.8s ease-out 0.4s both;
        font-size: 0.85rem;
    }
    .sidebar-header {
        font-size: 1.2rem; font-weight: 700; color: #3b82f6;
        margin-bottom: 1rem; animation: pulse 2s ease-in-out infinite;
    }
    .stButton button {
        background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
        border: none !important;
        animation: pulse 2s ease-in-out infinite !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem !important;
    }
    .stButton button:hover {
        animation: none !important;
        transform: scale(1.02) !important;
        box-shadow: 0 8px 30px rgba(59,130,246,0.3) !important;
    }
    .section-title {
        font-size: 1rem; font-weight: 700; color: #fff;
        margin: 1rem 0 0.5rem;
        animation: slideIn 0.5s ease-out;
    }
    .progress-container {
        background: rgba(255,255,255,0.05); border-radius: 20px; height: 12px; overflow: hidden; margin: 1rem 0;
    }
    .progress-bar {
        height: 100%; border-radius: 20px;
        transition: width 1.5s ease-out;
    }
    .input-group {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 0.75rem 0.75rem 0.25rem;
        margin-bottom: 0.75rem;
        border-left: 3px solid rgba(59,130,246,0.3);
    }
    .input-group-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #3b82f6;
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
    .stSidebar label { font-size: 0.85rem !important; font-weight: 500 !important; }
    .stSidebar .stSlider { padding-bottom: 0.25rem; }
    .stSidebar .stSelectbox { padding-bottom: 0.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">👥 HR Attrition Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict whether an employee is likely to leave the company</div>',
    unsafe_allow_html=True,
)

DATA_PATH = "data/hr_attrition.csv"
DATA_URL = "https://raw.githubusercontent.com/mragpavank/ibm-hr-analytics-attrition-dataset/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"


@st.cache_data
def load_hr_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        df = pd.read_csv(DATA_URL)
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df


@st.cache_resource
def train_hr_model(df):
    cat_cols = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]
    num_cols = ["Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
                "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction", "MonthlyIncome",
                "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating",
                "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
                "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
                "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]
    avail_num = [c for c in num_cols if c in df.columns]
    avail_cat = [c for c in cat_cols if c in df.columns]
    le_dict = {}
    for c in avail_cat:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        le_dict[c] = le
    feature_cols = avail_num + avail_cat
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    return model, scaler, le_dict, feature_cols, avail_num, avail_cat


df = load_hr_data()
with st.spinner("👥 Training model..."):
    model, scaler, le_dict, feature_cols, num_cols, cat_cols = train_hr_model(df.copy())
for c in cat_cols:
    if c in df.columns and c in le_dict:
        df[c] = le_dict[c].transform(df[c].astype(str))

if "hr_reset_key" not in st.session_state:
    st.session_state.hr_reset_key = 0

rk = st.session_state.hr_reset_key

hr_groups = [
    ("👤 Personal", [
        ("age", lambda k: st.sidebar.slider("🎂 Age", 18, 60, 32, key=f"hr_a_{k}")),
        ("marital_status", lambda k: st.sidebar.selectbox("💍 Marital Status", ["Single", "Married", "Divorced"], key=f"hr_m_{k}")),
        ("distance_from_home", lambda k: st.sidebar.slider("📍 Distance (km)", 1, 30, 10, key=f"hr_d_{k}")),
    ]),
    ("💼 Job", [
        ("job_role", lambda k: st.sidebar.selectbox("📋 Job Role", sorted(df["JobRole"].unique()), key=f"hr_jr_{k}")),
        ("department", lambda k: st.sidebar.selectbox("🏢 Department", sorted(df["Department"].unique()), key=f"hr_dp_{k}")),
        ("education_field", lambda k: st.sidebar.selectbox("🎓 Education", sorted(df["EducationField"].unique()), key=f"hr_ef_{k}")),
        ("overtime", lambda k: st.sidebar.selectbox("⏰ Overtime", ["No", "Yes"], key=f"hr_ot_{k}")),
        ("business_travel", lambda k: st.sidebar.selectbox("✈️ Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], key=f"hr_bt_{k}")),
    ]),
    ("💰 Compensation", [
        ("monthly_income", lambda k: st.sidebar.slider("💰 Monthly Income ($)", 1000, 20000, 5000, step=500, key=f"hr_mi_{k}")),
        ("total_working_years", lambda k: st.sidebar.slider("📅 Total Working Years", 0, 40, 10, key=f"hr_twy_{k}")),
        ("years_at_company", lambda k: st.sidebar.slider("🏠 Years at Company", 0, 40, 5, key=f"hr_yac_{k}")),
        ("years_in_current_role", lambda k: st.sidebar.slider("📌 Years in Current Role", 0, 20, 3, key=f"hr_ycr_{k}")),
        ("num_companies_worked", lambda k: st.sidebar.slider("🏢 Companies Worked", 0, 10, 2, key=f"hr_ncw_{k}")),
    ]),
    ("⭐ Satisfaction", [
        ("job_satisfaction", lambda k: st.sidebar.selectbox("😊 Job Satisfaction", [1, 2, 3, 4], key=f"hr_js_{k}")),
        ("work_life_balance", lambda k: st.sidebar.selectbox("⚖️ Work-Life Balance", [1, 2, 3, 4], key=f"hr_wlb_{k}")),
        ("environment_satisfaction", lambda k: st.sidebar.selectbox("🌿 Environment Satisfaction", [1, 2, 3, 4], key=f"hr_es_{k}")),
    ]),
]

hr_inputs = {}
for group_label, items in hr_groups:
    st.sidebar.markdown(
        f'<div class="input-group"><div class="input-group-label">{group_label}</div>',
        unsafe_allow_html=True,
    )
    for name, widget_fn in items:
        hr_inputs[name] = widget_fn(rk)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("---")

col_a, col_b = st.sidebar.columns(2)
predicted = False
with col_a:
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        predicted = True
with col_b:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.hr_reset_key += 1
        st.rerun()

st.sidebar.markdown(
    '<div class="sidebar-info">Fill in employee details and click Predict</div>',
    unsafe_allow_html=True,
)

if predicted:
    input_data = {
        "Age": hr_inputs["age"], "MonthlyIncome": hr_inputs["monthly_income"],
        "YearsAtCompany": hr_inputs["years_at_company"],
        "DistanceFromHome": hr_inputs["distance_from_home"],
        "OverTime": hr_inputs["overtime"], "JobRole": hr_inputs["job_role"],
        "JobSatisfaction": hr_inputs["job_satisfaction"],
        "WorkLifeBalance": hr_inputs["work_life_balance"],
        "EnvironmentSatisfaction": hr_inputs["environment_satisfaction"],
        "MaritalStatus": hr_inputs["marital_status"],
        "EducationField": hr_inputs["education_field"],
        "Department": hr_inputs["department"],
        "BusinessTravel": hr_inputs["business_travel"],
        "TotalWorkingYears": hr_inputs["total_working_years"],
        "YearsInCurrentRole": hr_inputs["years_in_current_role"],
        "NumCompaniesWorked": hr_inputs["num_companies_worked"],
        "DailyRate": 800, "Education": 3, "HourlyRate": 70, "JobInvolvement": 3,
        "JobLevel": 2, "MonthlyRate": 15000, "PercentSalaryHike": 15,
        "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1,
        "TrainingTimesLastYear": 3, "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3, "Gender": "Male", "EmployeeCount": 1,
        "Over18": "Y", "StandardHours": 80, "EmployeeNumber": 1,
    }
    for c in cat_cols:
        if c in le_dict:
            val = str(input_data.get(c, "Unknown"))
            if val not in le_dict[c].classes_:
                le_dict[c].classes_ = np.append(le_dict[c].classes_, val)
            input_data[c] = le_dict[c].transform([val])[0]

    input_df = pd.DataFrame([input_data])[feature_cols]
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0]
    pred = model.predict(input_scaled)[0]
    attrition_risk = proba[1]

    if attrition_risk < 0.3:
        risk_class = "risk-low"
        risk_color = "#22c55e"
        badge_text = "✅ Low Risk"
        badge_bg = "rgba(34,197,94,0.2)"
    elif attrition_risk < 0.6:
        risk_class = "risk-medium"
        risk_color = "#eab308"
        badge_text = "⚠️ Medium Risk"
        badge_bg = "rgba(234,179,8,0.2)"
    else:
        risk_class = "risk-high"
        risk_color = "#ef4444"
        badge_text = "🔴 High Risk"
        badge_bg = "rgba(239,68,68,0.2)"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="risk-card {risk_class}">
                <div class="risk-percent" style="color:{risk_color}">{attrition_risk:.1%}</div>
                <div class="risk-label" style="color:#aaa;">Attrition Risk</div>
                <div class="risk-badge" style="background:{badge_bg};color:{risk_color};">{badge_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="animation:fadeInUp 0.8s ease-out 0.2s both;">
                <div class="metric-box" style="animation-delay:0.2s">
                    <div class="metric-value">{model.score(scaler.transform(df[feature_cols].fillna(df[feature_cols].median())), df["Attrition"]):.1%}</div>
                    <div class="metric-label">Model Accuracy</div>
                </div>
                <div class="metric-box" style="animation-delay:0.4s;margin-top:1rem;">
                    <div class="metric-value">{"Yes" if pred == 1 else "No"}</div>
                    <div class="metric-label">Prediction: Will Leave?</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    progress_pct = int(attrition_risk * 100)
    bar_color = "#22c55e" if progress_pct < 30 else ("#eab308" if progress_pct < 60 else "#ef4444")
    st.markdown(
        f"""
        <div class="section-title">📊 Risk Level</div>
        <div class="progress-container">
            <div class="progress-bar" style="width:{progress_pct}%;background:linear-gradient(90deg,{bar_color},{bar_color}88);"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">🔑 Top Risk Factors</div>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        "Feature": feature_cols, "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True).tail(10)
    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ccc"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if pred == 1:
        st.markdown(
            '<div class="insight-box">⚠️ <strong>High attrition risk detected.</strong> Consider: salary review, '
            'improving work-life balance, career growth opportunities, and addressing overtime concerns.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="insight-box">✅ <strong>Low attrition risk.</strong> The employee appears satisfied with '
            'current conditions. Continue monitoring work-life balance and compensation.</div>',
            unsafe_allow_html=True,
        )
        st.balloons()
else:
    st.info("👈 Fill in the employee profile in the sidebar and click **Predict Attrition** to see results.")
    st.markdown(
        f"<div style='background:rgba(59,130,246,0.05);border-radius:16px;padding:1.25rem;margin-top:1rem;'>"
        f"<span style='color:#3b82f6;font-weight:700;'>{len(df):,}</span> "
        f"<span style='color:#888;'>employees in dataset · "
        f"<span style='color:#ef4444;'>{df['Attrition'].mean():.1%}</span> attrition rate</span></div>",
        unsafe_allow_html=True,
    )
