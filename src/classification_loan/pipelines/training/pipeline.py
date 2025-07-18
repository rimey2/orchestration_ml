from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data, train_model, evaluate_model, save_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs="encoded_data",  #  dataset d'encodage définie dans catalog.yml
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train",
                    "params:log_to_mlflow",
                    "params:experiment_id"],
            outputs=dict(model="model", mlflow_run_id="mlflow_run_id"),
            name="train_model_node"
            
        ),


        node(
            func=evaluate_model,
            inputs=["model", "X_test", "y_test", "params:log_to_mlflow"],
            outputs="evaluation_report",
            name="evaluate_model_node",
        ),
        node(
            func=save_model,
            inputs="model",
            outputs=None,
            name="save_model_node",
        ),
    ])
