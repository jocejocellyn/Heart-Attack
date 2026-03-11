import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load preprocess and model from MLflow
# Load preprocessor
scaler = joblib.load("artifacts/preprocessor.pkl")
model = joblib.load("artifacts/model.pkl")

def main():
    st.title('Machine Learning Heart Attack Prediction Model Deployment')

    age = st.number_input("Age", min_value=1, max_value=120, value=40) 
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0,1]) 
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3]) 
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120) 
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200) 
    fbs = st.selectbox("Fasting Blood Sugar >120 (0 = No, 1 = Yes)", [0,1]) 
    restecg = st.selectbox("Resting ECG (0-2)", [0,1,2]) 
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150) 
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0,1]) 
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0) 
    slope = st.selectbox("Slope of Peak Exercise ST (0-2)", [0,1,2]) 
    ca = st.selectbox("Number of Major Vessels (0-4)", [0,1,2,3,4]) 
    thal = st.selectbox("Thalassemia (0-3)", [0,1,2,3])
    
    if st.button('Make Prediction'):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        result = make_prediction(features)
        if result == 1: 
            st.error("High Risk of Heart Attack") 
        else: 
            st.success("Low Risk of Heart Attack")

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)
    return prediction[0]

if __name__ == '__main__':
    main()

