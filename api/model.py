import os
import pandas as pd
import joblib

class Model:
    def __init__(self):
        self.model = None
        try:
            self.load_model()
        except Exception as e:
            print(f"❌ Erreur lors du chargement du modèle : {e}")

    def load_model(self):
        model_path = "data/06_models/final_model.pkl"
        print(f"🔄 Chargement du modèle local : {model_path}")
        self.model = joblib.load(model_path)
        print("✅ Modèle chargé avec succès.")

    def predict(self, df: pd.DataFrame):
        return self.model.predict(df)
