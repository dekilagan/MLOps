# voicegenclass/assets/split_data.py
from dagster import asset, AssetIn
import pandas as pd
from sklearn.model_selection import train_test_split

@asset(ins={"engineered_features": AssetIn()})
def data_split(engineered_features: pd.DataFrame) -> dict:
    """
    Splits the engineered data into training and testing sets.
    
    Returns:
        A dictionary with keys: "X_train", "X_test", "y_train", "y_test".
    """
    X = engineered_features.drop("label", axis=1)
    y = engineered_features["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
