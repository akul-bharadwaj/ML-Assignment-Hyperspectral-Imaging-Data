from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import cloudpickle
import tensorflow as tf
import logging
from src.preprocessing import preprocess_new_data

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load pre-trained models
fnn_model = tf.keras.models.load_model("models/fnn_model.h5", compile=False)
xgb_model = joblib.load("models/xgb_model.pkl")
with open("models/ensemble_model.pkl", "rb") as f:
    ensemble_model = cloudpickle.load(f)
    
# Load dataset
DATA_PATH = "data/new_gen_data.csv"
df = pd.read_csv(DATA_PATH)
df.set_index("hsi_id", inplace=True)  # Set hsi_id as index for easy lookup

logger.info("Models and dataset loaded successfully.")

app = FastAPI()

@app.post("/predict")
def predict(data: dict):
    try:
        hsi_id = data["hsi_id"]
        logger.info(f"Received request for hsi_id: {hsi_id}")

        # Check if hsi_id exists in dataset
        if hsi_id not in df.index:
            logger.error(f"hsi_id '{hsi_id}' not found in dataset.")
            raise HTTPException(status_code=404, detail="hsi_id not found in dataset.")

        # Extract features for the given hsi_id
        features = df.loc[hsi_id].values.tolist()
        features_scaled = preprocess_new_data(features)

        # Get predictions from all models
        fnn_pred = fnn_model.predict(features_scaled)[0][0]
        xgb_pred = xgb_model.predict(features_scaled)[0]
        ensemble_pred = ensemble_model.predict(features_scaled)[0]

        logger.info(f"Predictions - FNN: {fnn_pred}, XGB: {xgb_pred}, Ensemble: {ensemble_pred}")

        return {
            "hsi_id": hsi_id,
            "fnn_prediction": float(fnn_pred),
            "xgb_prediction": float(xgb_pred),
            "ensemble_prediction": float(ensemble_pred),
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
