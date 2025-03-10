import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit interface
st.title('💳 Fraud Detection Prediction')
st.markdown("Enter values for all 30 features used in the model.")

# Feature input
input_features = []
for i in range(30):
    value = st.sidebar.number_input(f'Feature {i+1}', value=0.0)
    input_features.append(value)

# Prepare features
features = np.array(input_features).reshape(1, -1)
features_scaled = scaler.transform(features)

# Predict
if st.sidebar.button('Predict'):
    prediction = model.predict(features_scaled)
    result = '🚨 Fraudulent Transaction' if prediction[0] == 1 else '✅ Legitimate Transaction'
    st.success(f'The prediction is: **{result}**')
