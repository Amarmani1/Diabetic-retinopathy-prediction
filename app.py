import streamlit as st
import joblib
import numpy as np

# Load the brain of your project
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("👁️ Diabetic Retinopathy Detector")
st.write("Enter the patient details to check for risk.")

# Create the input boxes
age = st.number_input("Age", min_value=1, max_value=100, value=50)
systolic = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)

if st.button("Predict Now"):
    # Prepare data for the model
    features = np.array([[age, systolic, diastolic, chol]])
    features_scaled = scaler.transform(features)
    
    # Get result
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        st.error("Result: High Risk of Retinopathy")
    else:
        st.success("Result: Low Risk of Retinopathy")