import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
pipeline = joblib.load("resume_screening_.pkl")

st.set_page_config(page_title="AI Resume Screening", layout="centered")
st.title("AI Resume Screening")
st.write("Enter candidate details to predict the most suitable **Job Role**")

# Input fields
skills = st.text_area("Skills", placeholder="e.g. Python, Machine Learning, SQL")
education = st.text_input("Education", placeholder="e.g. B.Tech Computer Science")
certifications = st.text_input("Certifications", placeholder="e.g. AWS, Azure, Data Science")
experience = st.number_input("Experience (Years)", min_value=0, max_value=50, step=1)
projects = st.number_input("Projects Count", min_value=0, max_value=100, step=1)
ai_score = st.slider("AI Score (0-100)", 0, 100, 50)
salary = st.number_input("Salary Expectation ($)", min_value=0, step=1000)

# Predict button
if st.button("Predict Job Role"):
    # Create dataframe for prediction
    input_data = pd.DataFrame([{
        "Skills": skills,
        "Education": education,
        "Certifications": certifications,
        "Experience (Years)": experience,
        "Projects Count": projects,
        "AI Score (0-100)": ai_score,
        "Salary Expectation ($)": salary
    }])

    # Make prediction
    prediction = pipeline.predict(input_data)[0]

    st.success(f"🎯 Predicted Job Role: **{prediction}**")
