from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load("model/crops_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Crop label dictionary (reverse mapping)
crops_dict = {
    1: "Sugarcane", 2: "Mango", 3: "Cauliflower", 4: "Tomato", 5: "Maize",
    6: "Cabbage", 7: "Millet", 8: "Soybean", 9: "Lentil", 10: "Apple",
    11: "Wheat", 12: "Chili", 13: "Mustard", 14: "Rice", 15: "Onion",
    16: "Banana", 17: "Potato", 18: "Barley", 19: "Garlic", 20: "Pea"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Prepare input
        input_data = np.array([[N, P, K, rainfall, temperature, humidity, ph]])
        scaled_data = scaler.transform(input_data)

        # Predict
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        crop_name = crops_dict.get(prediction, "Unknown")

        return jsonify({'crop': crop_name, 'confidence': confidence})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
