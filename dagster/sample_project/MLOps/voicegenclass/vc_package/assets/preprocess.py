# voicegenclass/assets/preprocess.py
from dagster import asset, AssetIn
import pandas as pd
from sklearn.preprocessing import StandardScaler

@asset(ins={"voice_data": AssetIn()})
def preprocessed_data(voice_data: pd.DataFrame) -> pd.DataFrame:
    """Cleans and scales the data."""
    df = voice_data.dropna()
    features = df.drop("label", axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)
    df_scaled["label"] = df["label"].values
    return df_scaled
