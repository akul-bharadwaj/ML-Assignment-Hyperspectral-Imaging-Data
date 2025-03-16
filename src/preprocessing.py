import joblib
import numpy as np
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load pre-trained scaler
scaler = joblib.load("models/scaler.pkl")
logger.info("Scaler loaded successfully.")

def check_data_consistency(features):
    """
    Check for data inconsistencies:
    - Correct number of features
    - No missing values
    """
    expected_features = scaler.n_features_in_

    if len(features) != expected_features:
        logger.error(f"Expected {expected_features} features, but got {len(features)}.")
        raise ValueError(f"Expected {expected_features} features, but got {len(features)}.")
    
    if any(x is None or np.isnan(x) for x in features):
        logger.error("Input contains missing or NaN values.")
        raise ValueError("Input contains missing or NaN values.")
    
    logger.info("Data consistency check passed.")
    return np.array(features).reshape(1, -1)

def preprocess_new_data(features):
    """
    Perform data consistency checks and scale input features.
    """
    features_checked = check_data_consistency(features)
    scaled_data = scaler.transform(features_checked)
    logger.info("Features successfully scaled.")
    return scaled_data
