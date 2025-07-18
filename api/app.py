import pandas as pd
from flask import Flask, request, jsonify
from model import Model

app = Flask(__name__)
model = Model()

@app.route("/", methods=["GET"])
def home():
    print("🟢 GET / called")
    return "API is up", 200

@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    df = pd.read_json(body)
    results = [int(x) for x in model.predict(df).flatten()]
    return jsonify(results), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8899)
