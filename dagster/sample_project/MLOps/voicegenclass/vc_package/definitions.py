# vc_package/definitions.py
from dagster import repository
from vc_package.assets import (
    voice_data,
    preprocessed_data,
    engineered_features,
    data_split,
    trained_model,
    model_accuracy,
)

@repository
def vc_repo():
    return [
        voice_data,
        preprocessed_data,
        engineered_features,
        data_split,
        trained_model,
        model_accuracy,
    ]
