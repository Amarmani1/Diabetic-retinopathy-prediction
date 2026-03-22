import streamlit as st
import joblib
import numpy as np
import xgboost as xgb

# Load the model and scaler
try:
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or Scaler files not found.")

st.title("👁️ Diabetic Retinopathy Prediction (Binary)")

# Inputs
age = st.number_input("Age", min_value=1, max_value=100, value=50)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

if st.button("Predict"):
    # 1. Prepare data
    features = np.array([[age, systolic, diastolic, chol]])
    features_scaled = scaler.transform(features)
    
    # 2. Get the raw prediction (0 or 1)
    prediction = model.predict(features_scaled)[0]
    
    # 3. Display Yes/No based on 0/1 logic
    st.write(f"**Model Output (Class):** {prediction}")
    
    if prediction == 1:
        st.error("### Result: YES (1)")
        st.write("The model predicts a high risk of Diabetic Retinopathy.")
    else:
        st.success("### Result: NO (0)")
        st.write("The model predicts a low risk/no Diabetic Retinopathy.")