from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    try:
        # Safely parse values — even if sent as string or missing
        velocity = float(data.get('velocity') or 0)
        mass = float(data.get('mass') or 0)
        meteor_type = str(data.get('type_meteor') or 'generic')
        year = int(data.get('year') or 1950)

        # Dummy example prediction
        prediction = (velocity * 0.5) + (mass * 0.2) + (year % 100) + len(meteor_type)

        return jsonify({
            "predicted_impact_force": prediction,
            "velocity": velocity,
            "mass": mass,
            "meteor_type": meteor_type,
            "year": year
        })
    except Exception as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
