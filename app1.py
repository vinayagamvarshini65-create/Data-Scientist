import streamlit as st
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load("titanic_model.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details to predict survival")

# User Inputs
p_class = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=32.0)

# Encode sex
sex_encoded = le.transform([sex])[0]

# Create DataFrame
input_data = pd.DataFrame({
    "p_class": [p_class],
    "sex": [sex_encoded],
    "age": [age],
    "fare": [fare]
})

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("🎉 Passenger Survived")
    else:
        st.error("❌ Passenger Did Not Survive")
