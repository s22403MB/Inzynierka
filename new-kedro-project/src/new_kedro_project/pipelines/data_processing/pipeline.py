from kedro.pipeline import Pipeline, node, pipeline

from .nodes import process_patients, split_patients, patients_fit_model, rate_model, cross_val


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_patients,
                inputs="patients",
                outputs="processed_patients",
                name="process_patients_node",
            ),
            node(
                func=split_patients,
                inputs=["processed_patients", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            ),
            node(
                func=patients_fit_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="model",
                name="train_model_node"
            ),
            node(
                func=rate_model,
                inputs=["model", "X_test", "y_test"],
                outputs=None,
                name="model_rate_node"
            ),
            node(
                func=cross_val,
                inputs=["model", "X_train", "y_train", "params:model_options"],
                outputs=None,
                name="model_cross_val_node"
            ),
        ]
    )
