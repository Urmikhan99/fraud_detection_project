# app/app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    result = 'Fraud' if prediction[0] == 1 else 'Legitimate'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)

    model = joblib.load('D:/AI/fraud_detection_project/fraud_detection_model.pkl')

