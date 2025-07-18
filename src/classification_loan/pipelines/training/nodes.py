import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os


import pickle

# 1. Séparation train/test



def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = data.drop("loan_status", axis=1)
    y = data["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

# 2. Entraînement du modèle RandomForest avec hyperparams (modulable)

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    log_to_mlflow: bool = False,
    experiment_id: str = None
):
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [10, 30],
        'min_samples_split': [2, 9],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best params:", best_params)
    print("Best AUC CV:", best_score)

    
    run_id = ""
    if log_to_mlflow:
        mlflow.set_tracking_uri('http://127.0.0.1:5000')

        run = mlflow.start_run(experiment_id=experiment_id,nested = True)
        run_id = run.info.run_id
        # Log hyperparams et score
        mlflow.log_params(best_params)
        mlflow.log_metric("roc_auc_cv", best_score)
        print("log_metric")
        # Log du modèle
       ## mlflow.log_artifact(artifact_path="")
        mlflow.sklearn.log_model(best_model, "model")

        mlflow.end_run()

    return  dict(model=best_model, mlflow_run_id=run_id)


# 3. Prédictions & évaluation
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,log_to_mlflow: bool = False):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc_value = roc_auc_score(y_test, y_proba)
    print(f"AUC: {roc_auc_value:.4f}")

    if log_to_mlflow:
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('We will see if he gets that loan')
        plt.legend(loc="lower right")
        
        roc_path = "roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

        mlflow.log_artifact(roc_path)
    return report

# 4. Sauvegarde du modèle (si besoin)
def save_model(model) -> None:
    with open("data/06_models/final_model.pkl", "wb") as f:
        pickle.dump(model, f)


