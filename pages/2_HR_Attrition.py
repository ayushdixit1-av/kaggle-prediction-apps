import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")
st.title("👥 HR Employee Attrition Predictor")
st.markdown("Predict whether an employee is likely to leave the company based on their profile.")

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
with st.spinner("Training model..."):
    model, scaler, le_dict, feature_cols, num_cols, cat_cols = train_hr_model(df)

st.sidebar.header("👤 Employee Profile")

age = st.sidebar.slider("Age", 18, 60, 32)
monthly_income = st.sidebar.slider("Monthly Income ($)", 1000, 20000, 5000, step=500)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
distance_from_home = st.sidebar.slider("Distance from Home (km)", 1, 30, 10)
overtime = st.sidebar.selectbox("Overtime", ["No", "Yes"])
job_role = st.sidebar.selectbox("Job Role", sorted(df["JobRole"].unique()))
job_satisfaction = st.sidebar.selectbox("Job Satisfaction (1-4)", [1, 2, 3, 4])
work_life_balance = st.sidebar.selectbox("Work-Life Balance (1-4)", [1, 2, 3, 4])
environment_satisfaction = st.sidebar.selectbox("Environment Satisfaction (1-4)", [1, 2, 3, 4])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education_field = st.sidebar.selectbox("Education Field", sorted(df["EducationField"].unique()))
department = st.sidebar.selectbox("Department", sorted(df["Department"].unique()))
business_travel = st.sidebar.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
years_in_current_role = st.sidebar.slider("Years in Current Role", 0, 20, 3)
num_companies_worked = st.sidebar.slider("Companies Worked At", 0, 10, 2)

if st.sidebar.button("🔮 Predict Attrition", type="primary", use_container_width=True):
    input_data = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
        "DistanceFromHome": distance_from_home,
        "OverTime": overtime,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "EnvironmentSatisfaction": environment_satisfaction,
        "MaritalStatus": marital_status,
        "EducationField": education_field,
        "Department": department,
        "BusinessTravel": business_travel,
        "TotalWorkingYears": total_working_years,
        "YearsInCurrentRole": years_in_current_role,
        "NumCompaniesWorked": num_companies_worked,
        "DailyRate": 800,
        "Education": 3,
        "HourlyRate": 70,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "MonthlyRate": 15000,
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StockOptionLevel": 1,
        "TrainingTimesLastYear": 3,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 3,
        "Gender": "Male",
        "EmployeeCount": 1,
        "Over18": "Y",
        "StandardHours": 80,
        "EmployeeNumber": 1,
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

    col1, col2 = st.columns(2)
    with col1:
        attrition_risk = proba[1]
        color = "green" if attrition_risk < 0.3 else ("orange" if attrition_risk < 0.6 else "red")
        st.markdown(
            f"""
            <div style="text-align:center;padding:2rem;border-radius:1rem;background-color:{'#1b3a1b' if attrition_risk < 0.3 else '#3a3a1b' if attrition_risk < 0.6 else '#3a1b1b'}">
                <h1 style="color:{color};font-size:4rem;margin:0">{attrition_risk:.1%}</h1>
                <p style="color:{color};font-size:1.2rem">Attrition Risk</p>
                <p style="color:#aaa">Prediction: {"⚠️ Will Leave" if pred == 1 else "✅ Will Stay"}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("### 📊 Model Performance")
        acc = model.score(scaler.transform(df[feature_cols].fillna(df[feature_cols].median())), df["Attrition"])
        st.metric("Accuracy", f"{acc:.1%}")
        st.metric("Model", "Random Forest (balanced)")

    st.markdown("### 🔑 Top Risk Factors")
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=True).tail(10)

    fig = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Reds",
        title="Top 10 factors influencing attrition",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    if pred == 1:
        st.warning("⚠️ **High attrition risk.** Consider: salary review, better work-life balance, career growth opportunities.")
    else:
        st.success("✅ **Low attrition risk.** Employee appears satisfied with current conditions.")
else:
    st.info("👈 Fill in the employee profile in the sidebar and click **Predict Attrition**.")

    with st.expander("📋 About the Dataset"):
        attrition_rate = df["Attrition"].mean()
        st.markdown(
            f"""
            - **Rows:** {len(df):,} employees
            - **Features:** 35 attributes
            - **Attrition rate:** {attrition_rate:.1%}
            - **Dataset:** IBM HR Analytics (fictional)
            """
        )
