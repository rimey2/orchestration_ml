from kedro.pipeline import node, pipeline
from .nodes import encode_data

def create_pipeline(**kwargs):
    return pipeline([
        node(
            func=encode_data,
            inputs="raw_data",
            outputs="encoded_data",
            name="encode_data_node"
        )
    ])
