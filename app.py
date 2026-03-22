import streamlit as st
import joblib
import numpy as np
import xgboost as xgb

# Load model and scaler
try:
    model = joblib.load('xgboost_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Missing model files.")

st.title("👁️ Diabetic Retinopathy Prediction")

# User Inputs
age = st.number_input("Age", min_value=1, max_value=100, value=25)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

if st.button("Predict"):
    # 1. Process Input
    features = np.array([[age, systolic, diastolic, chol]])
    features_scaled = scaler.transform(features)
    
    # 2. Get Single Binary Result (0 or 1)
    prediction = int(model.predict(features_scaled)[0])
    
    # 3. Output exactly one clear result
    st.markdown("---")
    st.subheader("Prediction Result:")
    
    if prediction == 1:
        st.error(f"YES ({prediction})")
    else:
        st.success(f"NO ({prediction})")