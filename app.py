import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="PlaceWise AI",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #4CAF50;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.metric-card {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & SCALER
# =========================

try:
    model = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

except:
    st.error("❌ Model files not found!")
    st.stop()

# =========================
# SIDEBAR
# =========================

st.sidebar.title("📌 About")

st.sidebar.info(
    """
    PlaceWise AI predicts placement readiness
    using Machine Learning and intelligent
    skill analysis.
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("🚀 Features")

st.sidebar.write("✅ Placement Prediction")
st.sidebar.write("✅ Readiness Analysis")
st.sidebar.write("✅ Weakness Detection")
st.sidebar.write("✅ Personalized Suggestions")
st.sidebar.write("✅ Visual Analytics")
st.sidebar.write("✅ Explainable AI Insights")

# =========================
# MAIN TITLE
# =========================

st.title("🎯 PlaceWise AI")

st.subheader("AI-Powered Placement Readiness System")

st.markdown("""
Analyze placement readiness using Machine Learning,
skill evaluation, and intelligent recommendations.
""")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================

st.subheader("📥 Enter Student Details")

col1, col2 = st.columns(2)

with col1:

    technical_skills = st.slider(
        "Technical Skills Score",
        0.0,
        10.0,
        5.0
    )

    soft_skills = st.slider(
        "Soft Skills Score",
        0.0,
        10.0,
        5.0
    )

    aptitude = st.slider(
        "Aptitude Score",
        0.0,
        100.0,
        50.0
    )

    communication = st.slider(
        "Communication Score",
        0.0,
        10.0,
        5.0
    )

with col2:

    degree = st.slider(
        "Degree Percentage",
        0.0,
        100.0,
        60.0
    )

    projects = st.number_input(
        "Projects Count",
        min_value=0,
        max_value=20,
        value=2
    )

    backlogs = st.number_input(
        "Backlogs",
        min_value=0,
        max_value=10,
        value=0
    )

# =========================
# READINESS SCORE
# =========================

def calculate_readiness():

    score = (
        (technical_skills * 10) * 0.2 +
        (soft_skills * 10) * 0.2 +
        aptitude * 0.2 +
        (communication * 10) * 0.2 +
        degree * 0.2
    )

    return round(score, 2)

# =========================
# WEAKNESS DETECTION
# =========================

def detect_weakness():

    weaknesses = []

    if technical_skills < 6:
        weaknesses.append("Technical Skills")

    if soft_skills < 6:
        weaknesses.append("Soft Skills")

    if aptitude < 60:
        weaknesses.append("Aptitude")

    if communication < 6:
        weaknesses.append("Communication")

    if degree < 60:
        weaknesses.append("Academics")

    if projects < 2:
        weaknesses.append("Projects")

    if backlogs > 0:
        weaknesses.append("Backlogs")

    return weaknesses

# =========================
# RECOMMENDATIONS
# =========================

def generate_recommendations(weaknesses):

    recommendations = []

    for weakness in weaknesses:

        if weakness == "Technical Skills":
            recommendations.append(
                "Practice coding and build real-world projects"
            )

        elif weakness == "Soft Skills":
            recommendations.append(
                "Improve teamwork and communication skills"
            )

        elif weakness == "Aptitude":
            recommendations.append(
                "Practice aptitude and reasoning daily"
            )

        elif weakness == "Communication":
            recommendations.append(
                "Participate in mock interviews and presentations"
            )

        elif weakness == "Academics":
            recommendations.append(
                "Improve academic performance and consistency"
            )

        elif weakness == "Projects":
            recommendations.append(
                "Build practical and portfolio-worthy projects"
            )

        elif weakness == "Backlogs":
            recommendations.append(
                "Clear academic backlogs quickly"
            )

    if len(recommendations) == 0:
        recommendations.append(
            "Excellent profile! Continue improving consistently."
        )

    return recommendations

# =========================
# EXPLAINABLE AI
# =========================

def explain_prediction():

    explanations = []

    if technical_skills >= 7:
        explanations.append(
            "Strong technical skills improved placement chances."
        )

    if aptitude >= 70:
        explanations.append(
            "Good aptitude score positively impacted prediction."
        )

    if communication >= 7:
        explanations.append(
            "Communication skills boosted readiness."
        )

    if projects >= 3:
        explanations.append(
            "Project experience strengthened the profile."
        )

    if backlogs > 0:
        explanations.append(
            "Backlogs negatively affected prediction."
        )

    return explanations

# =========================
# ANALYZE BUTTON
# =========================

if st.button("🚀 Analyze Placement Readiness"):

    # =========================
    # INPUT ARRAY
    # =========================

    input_data = np.array([[
        technical_skills,
        soft_skills,
        aptitude,
        communication,
        degree,
        projects,
        backlogs
    ]])

    # =========================
    # SCALE INPUT
    # =========================

    input_scaled = scaler.transform(input_data)

    # =========================
    # PREDICTION
    # =========================

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    # =========================
    # READINESS ANALYSIS
    # =========================

    readiness_score = calculate_readiness()

    weaknesses = detect_weakness()

    recommendations = generate_recommendations(
        weaknesses
    )

    explanations = explain_prediction()

    # =========================
    # READINESS LEVEL
    # =========================

    if readiness_score >= 75 and len(weaknesses) <= 1:
        readiness_level = "Placement Ready"

    elif readiness_score >= 60:
        readiness_level = "Moderately Ready"

    else:
        readiness_level = "Needs Improvement"

    # =========================
    # PROGRESS BAR
    # =========================

    st.progress(int(readiness_score))

    st.markdown("---")

    # =========================
    # PREDICTION RESULT
    # =========================

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Likely To Be Placed")

    else:
        st.error("❌ Placement Needs Improvement")

    # =========================
    # CONFIDENCE LEVEL
    # =========================

    if probability >= 0.8:
        st.success("High Placement Confidence")

    elif probability >= 0.6:
        st.warning("Moderate Placement Confidence")

    else:
        st.error("Low Placement Confidence")

    # =========================
    # METRICS
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Readiness Score",
        readiness_score
    )

    col2.metric(
        "Placement Probability",
        f"{round(probability * 100, 2)}%"
    )

    col3.metric(
        "Projects",
        projects
    )

    st.info(f"📈 Readiness Level: {readiness_level}")

    st.markdown("---")

    # =========================
    # VISUAL ANALYTICS
    # =========================

    st.subheader("📉 Skill Analysis")

    skills = [
        "Technical",
        "Soft Skills",
        "Aptitude",
        "Communication",
        "Academics"
    ]

    scores = [
        technical_skills * 10,
        soft_skills * 10,
        aptitude,
        communication * 10,
        degree
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(skills, scores)

    ax.set_ylim(0, 100)

    ax.set_ylabel("Scores")

    st.pyplot(fig)

    # =========================
    # WEAKNESSES
    # =========================

    st.subheader("⚠️ Weakness Detection")

    if weaknesses:

        for weakness in weaknesses:
            st.write("•", weakness)

    else:
        st.success("No major weaknesses detected")

    # =========================
    # RECOMMENDATIONS
    # =========================

    st.subheader("💡 Personalized Recommendations")

    for recommendation in recommendations:
        st.write("✅", recommendation)

    # =========================
    # EXPLAINABLE AI
    # =========================

    st.subheader("🧠 Why This Prediction?")

    if explanations:

        for explanation in explanations:
            st.write("📌", explanation)

    else:
        st.write(
            "Balanced profile detected by the ML model."
        )

    # =========================
    # KEY FACTORS
    # =========================

    st.subheader("📌 Key Placement Factors")

    st.write("""
    Important factors affecting placement prediction:

    - Technical Skills
    - Aptitude Performance
    - Communication Skills
    - Academic Consistency
    - Project Experience
    - Backlogs
    """)

    # =========================
    # DOWNLOAD REPORT
    # =========================

    st.subheader("📥 Download Report")

    report = f"""
PLACEWISE AI REPORT

Placement Prediction:
{"Likely To Be Placed" if prediction == 1 else "Needs Improvement"}

Placement Probability:
{round(probability * 100, 2)}%

Readiness Score:
{readiness_score}

Readiness Level:
{readiness_level}

Weaknesses:
{', '.join(weaknesses)}

Recommendations:
{', '.join(recommendations)}
"""

    st.download_button(
        label="📄 Download Analysis Report",
        data=report,
        file_name="placement_report.txt",
        mime="text/plain"
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Developed using Machine Learning and Streamlit 🚀"
)