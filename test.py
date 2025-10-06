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
    "mass (g)", "year", "recclass_encoded", "velocity_km_s",
    "mass_log", "velocity_squared", "momentum",
    "kinetic_energy", "year_modern", "mass_velocity_interaction"
]

# Uncertainty values (from training)
lat_std = 1.23  # replace with your actual computed value
long_std = 1.45


@app.route('/')
def home():
    return jsonify({"status": "Meteorite AI model running!!!"})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Parse and validate inputs
        velocity = float(data.get('velocity', 0))
        mass = float(data.get('mass', 0))
        meteor_type = str(data.get('type_meteor', 'H5'))
        year = int(data.get('year', 1950))

        # Encode meteorite class (recclass)
        try:
            recclass_encoded = le.transform([meteor_type])[0]
        except Exception:
            recclass_encoded = le.transform([le.classes_[0]])[0]  # fallback to known class

        # Compute engineered features
        mass_log = np.log1p(mass)
        velocity_squared = velocity ** 2
        momentum = mass * velocity
        kinetic_energy = 0.5 * mass * (velocity ** 2)
        year_modern = 1 if year >= 2000 else 0
        mass_velocity_interaction = mass_log * velocity

        # Prepare input for model
        sample = pd.DataFrame([[mass, year, recclass_encoded, velocity,
                                mass_log, velocity_squared, momentum,
                                kinetic_energy, year_modern, mass_velocity_interaction]],
                              columns=X_columns)

        sample_scaled = scaler.transform(sample)
        prediction = xgb_model.predict(sample_scaled)[0]

        # Build result
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

    except Exception as e:
        # Catch any runtime errors and return JSON
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

