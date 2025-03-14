from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI()

# Define the input schema. Note: we use Field with alias for 'sp.ent'
class VoiceFeatures(BaseModel):
    meanfreq: float
    sd: float
    median: float
    Q25: float
    Q75: float
    IQR: float
    skew: float
    kurt: float
    sp_ent: float = Field(..., alias="sp.ent")
    sfm: float
    mode: float
    centroid: float
    meanfun: float
    minfun: float
    maxfun: float
    meandom: float
    mindom: float
    maxdom: float
    dfrange: float
    modindx: float

# Instead of using a model alias, load the model via its run ID.
# Use actual run ID from training run.
RUN_ID = "YOUR_RUN_ID_HERE"
model_uri = f"runs:/134278d1da014112acb561d8da3e01cf/model"

# Load the model using MLflow's pyfunc interface.
model = mlflow.pyfunc.load_model(model_uri)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Voice Gender Classification API"}

@app.post("/predict")
def predict_voice_gender(features: VoiceFeatures):
    # Convert input using the alias so that "sp.ent" is preserved.
    input_data = features.dict(by_alias=True)
    df = pd.DataFrame([input_data])
    predictions = model.predict(df)
    return {"prediction": predictions.tolist()}
