import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fitness & Health Risk Prediction System",page_icon="💪",layout="wide")

#load saved models
model = pickle.load(open("health_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

df = pd.read_csv("cleaned_health_data.csv")

#custom csv
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.metric-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    text-align: center;
}

.big-font {
    font-size:25px !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

#sidebar
st.sidebar.title("💪 Health Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📊 Health Dashboard",
        "🧠 Risk Analysis",
        "💡 Smart Recommendations",
        "📈 Insights"
    ]
)

#homepage
if page == "🏠 Home":
    st.title("💪 Fitness & Health Risk Prediction System")
    st.markdown("---")
    st.subheader("AI-Based Lifestyle & Health Analytics")

    st.write("""
    This intelligent healthcare analytics system uses
    machine learning and lifestyle indicators to analyze:

    ✅ Diabetes Risk

    ✅ Lifestyle Quality

    ✅ Fitness Condition

    ✅ Wellness Indicators

    ✅ Health Patterns

    The dashboard provides interactive visual analytics
    and personalized recommendations for healthier living.
    """)

    st.image(
        "https://images.unsplash.com/photo-1517836357463-d25dfeac3438",
        use_container_width=True
    )

#dashboard
elif page == "📊 Health Dashboard":
    st.title("📊 Personal Health Dashboard")
    st.markdown("---")
    st.subheader("Lifestyle Controls")

    col1, col2 = st.columns(2)
    with col1:
        activity = st.slider("🏃 Physical Activity Level",0,10,5)

        sleep = st.slider("😴 Sleep Quality",1,10,7)

        stress = st.slider("🧠 Stress Level",1,10,4)

        bmi = st.slider("⚖️ BMI Level",15,40,24)

    with col2:
        smoking = st.toggle("🚬 Smoking Habit")
        alcohol = st.toggle("🍺 Alcohol Consumption")
        exercise = st.toggle("💪 Regular Exercise")
        walking = st.toggle("🚶 Difficulty Walking")

#health score
    score = 100
    score -= stress * 3
    score -= bmi - 20
    if smoking:
        score -= 15
    if alcohol:
        score -= 10
    if not exercise:
        score -= 15
    if walking:
        score -= 10
    score += activity * 2
    score += sleep
    score = max(0, min(score, 100))

#health category
    if score >= 85:
        category = "🔥 Fitness Enthusiast"
    elif score >= 70:
        category = "✅ Balanced Lifestyle"
    elif score >= 50:
        category = "⚠️ Health Needs Attention"
    else:
        category = "🚨 High Risk Lifestyle"


    st.markdown("---")
    st.subheader("Health Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Health Score",f"{score}/100")

    with col2:
        st.metric("Lifestyle Status",category)

    with col3:
        st.metric("Sleep Quality",f"{sleep}/10")

    st.subheader("Overall Wellness")
    st.progress(int(score))

#risk analysis
elif page == "🧠 Risk Analysis":
    st.title("🧠 AI Risk Analysis")
    st.markdown("---")
    st.subheader("Health Risk Overview")

    # Dummy example values
    BMI = 28
    Smoking = 1
    AlcoholDrinking = 0
    Stroke = 0
    PhysicalHealth = 5
    MentalHealth = 4
    DiffWalking = 0
    Sex = 1
    Age = 7
    PhysicalActivity = 1
    SleepTime = 7


    input_data = np.array([
        [
            BMI,
            Smoking,
            AlcoholDrinking,
            Stroke,
            PhysicalHealth,
            MentalHealth,
            DiffWalking,
            Sex,
            Age,
            PhysicalActivity,
            SleepTime
        ]
    ])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    #risk output
    if prediction[0] == 0:
        st.success("✅ Low Diabetes Risk")
        risk_percent = 25

    elif prediction[0] == 1:
        st.warning("⚠️ Prediabetes Risk Detected")
        risk_percent = 60

    else:
        st.error("🚨 High Diabetes Risk")
        risk_percent = 85

    st.subheader("Risk Meter")
    st.progress(risk_percent)
    st.write(f"Risk Probability: {risk_percent}%")

#recommendations

elif page == "💡 Smart Recommendations":
    st.title("💡 Personalized Health Recommendations")
    st.markdown("---")

    recommendations = [
        "1. Exercise at least 30 minutes daily",

        "2. Maintain proper sleep schedule",

        "3. Reduce sugar intake",

        "4. Drink more water",

        "5. Avoid smoking",

        "6. Reduce stress levels",

        "7. Monitor BMI regularly",

        "8. Include fruits and vegetables in diet"
    ]

    for item in recommendations:
        st.write(item)

#insights

elif page == "📈 Insights":
    st.title("📈 Health Insights & Analytics")
    st.markdown("---")

    #bmi
    st.subheader("BMI Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["BMI"],kde=True,ax=ax)
    st.pyplot(fig)

    st.subheader("Mental Health Analysis")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["MentHlth"],kde=True,ax=ax)
    st.pyplot(fig)

    st.subheader("Diabetes Category Distribution")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.countplot(x="Diabetes",data=df,ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(),cmap="coolwarm",ax=ax)
    st.pyplot(fig)