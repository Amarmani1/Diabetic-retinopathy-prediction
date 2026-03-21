import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb  # Make sure to import xgboost

# 1. Load the XGBoost model and the scaler
# Ensure these filenames match the ones you saved during training
try:
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Model files not found. Please upload xgboost_model.pkl and scaler.pkl")

st.title("👁️ Diabetic Retinopathy Risk Predictor")
st.write("This system uses a **Gradient Boosting (XGBoost)** model to predict risk levels.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, value=50)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

if st.button("Predict Now"):
    # Prepare data
    features = np.array([[age, systolic, diastolic, chol]])
    features_scaled = scaler.transform(features)
    
    # XGBoost prediction
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        st.error("Result: High Risk of Retinopathy")
    else:
        st.success("Result: Low Risk of Retinopathy")

st.info("Note: This model was analysed and tested to ensure high predictive reliability.")