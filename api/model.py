import os
import mlflow
import joblib
import pandas as pd

from mlflow.tracking import MlflowClient

# Config depuis .env
mlflow.set_tracking_uri(os.getenv("MLFLOW_SERVER"))

class Model:
    def __init__(self):
        self.model = None
        self.transform_pipeline = None

        try:
            self.load_model()
        except Exception as e:
            print(f"❌ ERREUR LORS DU CHARGEMENT COMPLET DU MODELE/PREPROCESS : {e}")

    def load_model(self):
        print("🔄 Tentative de chargement du modèle MLflow...")
        try:
            self.model = mlflow.sklearn.load_model(
                f"models:/{os.getenv('MLFLOW_REGISTRY_NAME')}@{os.getenv('ENV')}"
            )
            print("✅ Modèle MLflow chargé.")
        except Exception as e:
            print(f"❌ ERREUR MLflow : {e}")
            raise

        print("🔄 Tentative de chargement du pipeline...")
        try:
            self.transform_pipeline = joblib.load("data/04_feature/transform_pipeline.pkl")
            print("✅ Pipeline chargé.")
        except Exception as e:
            print(f"❌ ERREUR Pipeline : {e}")
            raise

