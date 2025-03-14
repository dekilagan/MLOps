# voicegenclass/assets/feature_engineering.py
from dagster import asset, AssetIn
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

@asset(ins={"preprocessed_data": AssetIn()})
def engineered_features(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on preprocessed voice data.
    
    Steps:
    - Generates polynomial interaction features (interaction-only).
    - Reduces dimensionality using PCA (e.g., to 10 components).
    
    Returns:
        A DataFrame with engineered features along with the target label.
    """
    # Separate features and target
    features = preprocessed_data.drop("label", axis=1)
    target = preprocessed_data["label"]
    
    # Generate polynomial interaction features
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(features)
    
    # Reduce dimensionality using PCA
    pca = PCA(n_components=10)
    pca_features = pca.fit_transform(poly_features)
    
    # Convert to DataFrame and attach target
    engineered_df = pd.DataFrame(pca_features, columns=[f"pca_{i}" for i in range(10)])
    engineered_df["label"] = target.values
    return engineered_df
