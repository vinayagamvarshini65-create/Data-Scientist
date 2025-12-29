import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("insurance_model.pkl")

st.title("Insurance Prediction App")
st.write("Predict whether a person will buy insurance based on age")

# User input
age = st.slider("Select Age", 18, 60, 30)

# Prediction
prediction = model.predict([[age]])
probability = model.predict_proba([[age]])[0][1]

# Output
if prediction[0] == 1:
    st.success("✅ Will Buy Insurance")
else:
    st.error("❌ Will NOT Buy Insurance")

st.write(f"**Probability:** {probability:.2f}")
