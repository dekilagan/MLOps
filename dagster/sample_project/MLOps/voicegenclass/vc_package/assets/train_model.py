# voicegenclass/assets/train_model.py
from dagster import asset, AssetIn
from sklearn.ensemble import RandomForestClassifier

@asset(ins={"data_split": AssetIn()})
def trained_model(data_split: dict) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier on the training data.
    
    Returns:
        A trained RandomForestClassifier model.
    """
    X_train = data_split["X_train"]
    y_train = data_split["y_train"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
