from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os
from automl_engine import AutoMLPipeline

app = FastAPI()

@app.post("/train/")
async def train_model(algorithm: str, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    automl = AutoMLPipeline(algorithm)
    automl.fit(X, y)
    
    model_filename = f"model_{algorithm}.pkl"
    automl.save_model(model_filename)

    return {"message": f"Model trained and saved as {model_filename}"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    model_path = "model_h2o.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    return {"error": "Model not found"}
