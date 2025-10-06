from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# ======== Load trained components ========
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Reference columns (same order as during training)
X_columns = [
    "mass", "year", "recclass", "velocity",
    "mass_log", "velocity_squared", "momentum",
    "kinetic_energy", "year_modern", "mass_velocity_interaction"
]

# Uncertainty values (from training)
lat_std = 1.23  # replace with your actual computed value
long_std = 1.45

@app.route('/')
def home():
    return jsonify({"status": "Meteorite AI model running ✅"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        grams = float(data.get("grams"))
        year = int(data.get("year"))
        recclass = str(data.get("recclass"))
        velocity = float(data.get("velocity"))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    # Encode recclass
    try:
        recclass_encoded = le.transform([recclass])[0]
    except:
        recclass_encoded = le.transform([le.classes_[0]])[0]  # fallback

    # Compute engineered features
    mass_log = np.log1p(grams)
    velocity_squared = velocity ** 2
    momentum = grams * velocity
    kinetic_energy = 0.5 * grams * (velocity ** 2)
    year_modern = 1 if year >= 2000 else 0
    mass_velocity_interaction = mass_log * velocity

    sample = pd.DataFrame([[
        grams, year, recclass_encoded, velocity,
        mass_log, velocity_squared, momentum, kinetic_energy,
        year_modern, mass_velocity_interaction
    ]], columns=X_columns)

    sample_scaled = scaler.transform(sample)
    prediction = xgb_model.predict(sample_scaled)[0]

    result = {
        "latitude": float(prediction[0]),
        "longitude": float(prediction[1]),
        "velocity": velocity,
        "uncertainty": {
            "latitude_std": lat_std,
            "longitude_std": long_std
        }
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

