# vc_package/assets/__init__.py
from .load_data import voice_data
from .preprocess import preprocessed_data
from .feature_engineering import engineered_features
from .split_data import data_split
from .train_model import trained_model
from .evaluate import model_accuracy

__all__ = [
    "voice_data",
    "preprocessed_data",
    "engineered_features",
    "data_split",
    "trained_model",
    "model_accuracy",
]
