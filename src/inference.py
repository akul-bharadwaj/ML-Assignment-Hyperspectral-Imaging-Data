import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from src.preprocessing import preprocess_new_data

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load the trained FNN model
fnn_model = tf.keras.models.load_model("models/fnn_model.h5", compile=False)

# Load dataset
DATA_PATH = "data/new_gen_data.csv"
df = pd.read_csv(DATA_PATH)
df.set_index("hsi_id", inplace=True)

logger.info("FNN Model and dataset loaded successfully.")

def predict(hsi_id):
    """
    Predict DON concentration using the FNN model for a given hsi_id.
    """
    try:
        logger.info(f"Fetching data for hsi_id: {hsi_id}")

        if hsi_id not in df.index:
            logger.error(f"hsi_id '{hsi_id}' not found in dataset.")
            return {"error": "hsi_id not found in dataset."}

        # Extract features
        features = df.loc[hsi_id].values.tolist()
        features_scaled = preprocess_new_data(features)

        # Get predictions
        fnn_pred = fnn_model.predict(features_scaled)[0][0]

        logger.info(f"Prediction - FNN: {fnn_pred}")

        return {
            "hsi_id": hsi_id,
            "fnn_prediction": float(fnn_pred),
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": "Prediction failed due to an internal error."}

if __name__ == "__main__":
    sample_hsi_id = "imagoai_corn_525"
    predictions = predict(sample_hsi_id)
    print("Predictions:", predictions)
