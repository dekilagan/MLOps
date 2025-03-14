# voicegenclass/assets/evaluate.py
from dagster import asset, AssetIn
from sklearn.metrics import accuracy_score

@asset(ins={"data_split": AssetIn(), "trained_model": AssetIn()})
def model_accuracy(data_split: dict, trained_model) -> float:
    """
    Evaluates the trained model on the test set and returns the accuracy.
    """
    X_test = data_split["X_test"]
    y_test = data_split["y_test"]
    y_pred = trained_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
