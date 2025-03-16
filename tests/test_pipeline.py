import unittest
import joblib
import cloudpickle
import pandas as pd
import numpy as np
import logging
from src.preprocessing import preprocess_new_data
from src.api import app
from fastapi.testclient import TestClient
import tensorflow as tf

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load Models
fnn_model = tf.keras.models.load_model("models/fnn_model.h5", compile=False)
xgb_model = joblib.load("models/xgb_model.pkl")
with open("models/ensemble_model.pkl", "rb") as f:
    ensemble_model = cloudpickle.load(f)

# Load dataset
DATA_PATH = "data/new_gen_data.csv"
df = pd.read_csv(DATA_PATH)
df.set_index("hsi_id", inplace=True)

# Test Client for FastAPI
client = TestClient(app)

class TestPipeline(unittest.TestCase):

    def test_preprocessing(self):
        """Test if preprocessing correctly scales input."""
        sample_input = [0.5] * 448  # Example with 448 features
        processed_data = preprocess_new_data(sample_input)
        self.assertEqual(processed_data.shape, (1, len(sample_input)))
        logger.info("Preprocessing test passed.")

    def test_hsi_id_extraction(self):
        """Test if hsi_id-based feature extraction works correctly."""
        test_hsi_id = df.index[0]  # Select the first available hsi_id
        features = df.loc[test_hsi_id].values.tolist()
        self.assertEqual(len(features), 448)  # Ensure correct number of features
        logger.info(f"Feature extraction test passed for hsi_id: {test_hsi_id}")

    def test_fnn_model(self):
        """Test if FNN model makes a valid prediction."""
        test_hsi_id = df.index[0]
        features = preprocess_new_data(df.loc[test_hsi_id].values.tolist())
        prediction = fnn_model.predict(features)[0][0]
        self.assertTrue(isinstance(prediction, (float, np.float32, np.float64)))
        logger.info(f"FNN model test passed for hsi_id: {test_hsi_id}, Prediction: {prediction}")

    def test_xgb_model(self):
        """Test if XGBoost model makes a valid prediction."""
        test_hsi_id = df.index[0]
        features = preprocess_new_data(df.loc[test_hsi_id].values.tolist())
        prediction = xgb_model.predict(features)[0]
        self.assertTrue(isinstance(prediction, (float, np.float32, np.float64)))
        logger.info(f"XGBoost model test passed for hsi_id: {test_hsi_id}, Prediction: {prediction}")

    def test_ensemble_model(self):
        """Test if Ensemble model makes a valid prediction."""
        test_hsi_id = df.index[0]
        features = preprocess_new_data(df.loc[test_hsi_id].values.tolist())
        prediction = ensemble_model.predict(features)[0]
        self.assertTrue(isinstance(prediction, (float, np.float32, np.float64)))
        logger.info(f"Ensemble model test passed for hsi_id: {test_hsi_id}, Prediction: {prediction}")

    def test_api_response(self):
        """Test if FastAPI returns valid responses for all models."""
        test_hsi_id = df.index[0]
        response = client.post("/predict", json={"hsi_id": test_hsi_id})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("fnn_prediction", data)
        self.assertIn("xgb_prediction", data)
        self.assertIn("ensemble_prediction", data)

        logger.info(f"API response test passed for hsi_id: {test_hsi_id}, Response: {data}")

if __name__ == "__main__":
    unittest.main()
