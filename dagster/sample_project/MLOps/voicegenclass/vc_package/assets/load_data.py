# vc_package/assets/load_data.py
from dagster import asset
import pandas as pd

@asset
def voice_data() -> pd.DataFrame:
    """Loads the raw voice data from a CSV file."""
    df = pd.read_csv("data/voice.csv")
    return df
