import pandas as pd
from flask import Flask, request, jsonify
from model import Model

app = Flask(__name__)
model = Model()

@app.route("/", methods=["GET"])
def home():
    return "API is up", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        print(f" Reçu : {body}")

        if isinstance(body, dict):
            df = pd.DataFrame([body])
        elif isinstance(body, list):
            df = pd.DataFrame(body)
        else:
            return jsonify({"error": "Format JSON invalide. Attendu : dict ou liste de dicts."}), 400

        print(f" DataFrame pour prédiction :\n{df.head()}")
        results = model.predict(df)
        return jsonify(results.tolist()), 200

    except Exception as e:
        print(f" Erreur dans /predict : {e}")
        return jsonify({"error": str(e)}), 500
