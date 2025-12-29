import streamlit as st
import pandas as pd
import joblib

# Load PKL files
model = joblib.load('exam_score_model.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('feature_selector.pkl')

st.title("🎓 Exam Score Prediction App")

st.write("Enter student details to predict exam score")

# User inputs
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0)

gender = st.selectbox("Gender", ["Male", "Female"])
parent_edu = st.selectbox("Parental Education", ["High School", "Bachelor", "Master"])
lunch = st.selectbox("Lunch Type", ["Standard", "Free"])
test_prep = st.selectbox("Test Preparation", ["None", "Completed"])

# Manual encoding (same as training)
gender = 1 if gender == "Male" else 0
parent_edu = {"High School": 0, "Bachelor": 1, "Master": 2}[parent_edu]
lunch = 1 if lunch == "Standard" else 0
test_prep = 1 if test_prep == "Completed" else 0

# Predict button
if st.button("Predict Exam Score"):
    input_df = pd.DataFrame([[
        study_hours, attendance, sleep_hours, previous_score,
        gender, parent_edu, lunch, test_prep
    ]], columns=[
        'Study_Hours', 'Attendance', 'Sleep_Hours', 'Previous_Score',
        'Gender', 'Parental_Education', 'Lunch_Type', 'Test_Preparation'
    ])

    scaled = scaler.transform(input_df)
    selected = selector.transform(scaled)
    prediction = model.predict(selected)

    st.success(f"📊 Predicted Exam Score: {prediction[0]:.2f}")
