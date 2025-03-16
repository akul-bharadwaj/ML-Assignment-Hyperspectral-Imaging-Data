# ML-Assignment-Hyperspectral-Imaging-Data

## Project Overview
This project provides a deployed machine learning pipeline to predict DON (Deoxynivalenol) concentration in corn samples using hyperspectral data. The pipeline includes:
- Data Preprocessing (with data consistency checks)
- Model Training (Jupyter Notebook)
- Model Inference (Pre-trained Feedforward Neural Network)
- API Deployment using FastAPI
- Containerization with Docker
- Unit Testing for Core Functionalities
- Data Generation Script for Simulated New Data
- Streamlit Web App for Real-Time Predictions (Bonus)

---

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/akul-bharadwaj/ML-Assignment-Hyperspectral-Imaging-Data
cd ML-Assignment-Hyperspectral-Imaging-Data
```

### 2. Create a Virtual Environment
It is recommended to create a virtual environment before installing dependencies.

#### **For Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### **For Windows PowerShell:**
```powershell
python -m venv venv
venv\Scripts\Activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run FastAPI Server
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## Generating New Data
A script `generate_new_data.py` is provided to create a synthetic dataset based on the existing data for inference.
This script:
- Generates `hsi_id` values from `imagoai_corn_500` to `imagoai_corn_549`.
- Selects random existing rows from `MLE-Assignment.csv` and applies slight variations to generate new feature values.
- Saves the newly generated dataset as `data/new_gen_data.csv`.

## Running Inference Locally
You can run `inference.py` to generate predictions locally for a given `hsi_id`:
```bash
python src/inference.py --hsi_id imagoai_corn_525
```
This will output:
```json
{
  "hsi_id": "imagoai_corn_525",
  "fnn_prediction": 123.45,
}
```

## Send a Prediction Request

#### **For Linux/macOS (cURL Command)**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"hsi_id": "imagoai_corn_525"}'
```

#### **For Windows PowerShell**
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"hsi_id": "imagoai_corn_525"}'
```

The response will contain predictions from:
- **Feedforward Neural Network (fnn_prediction)**

---

## Running Unit Tests
To validate core functionalities like preprocessing, model inference, and API response, run:
```bash
python -m unittest discover tests/
```

---

## Docker Deployment

### 1. Build the Docker Image
```bash
docker build -t don-prediction .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 don-prediction
```

### 3. Test the API

#### **For Linux/macOS (cURL Command)**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"hsi_id": "imagoai_corn_525"}'
```

#### **For Windows PowerShell**
```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict" -Method POST -Headers @{"Content-Type"="application/json"} -Body '{"hsi_id":"imagoai_corn_525"}'
```

The response will contain three predictions:
```json
{
  "fnn_prediction": 123.45,
}
```

---

## Dockerfile
```
FROM python:3.11
WORKDIR /app
COPY . /app
COPY models/fnn_model.h5 /app/models/
COPY data/ /app/data/
COPY generate_new_data.py /app/
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## (BONUS) Run the Streamlit Web App

The Streamlit app allows users to upload a CSV file containing spectral data and receive real-time predictions.

```bash
streamlit run streamlit_app.py
```

Upload a CSV file and select an `hsi_id` for prediction. The app will send a request to the FastAPI backend and display the predicted result.

---

### <b>Note</b>: For some of the key data analysis and model performance visualizations, I have used [Plotly](https://plotly.com/) to make the plots interactive.

## Repository Structure
```
├── models/                     # Saved pre-trained models
│   ├── fnn_model.h5            # Trained Feedforward Neural Network
│   ├── xgb_model.pkl           # Trained XGBoost Model
│   ├── ensemble_model.pkl      # Ensemble Model
|   ├── scaler.pkl
├── data/                       # Jupyter Notebook for model training
│   ├── MLE-Assignment.csv      # Dataset for model training and evaluation
│   ├── new_gen_data.csv        # New generated dataset for inference
├── visualizations-interactive/ # Interactive visualizations for some of the key data analysis and model performance
├── src/
│   ├── preprocessing.py        # Data preprocessing (scaling + consistency checks)
│   ├── api.py                  # FastAPI implementation (returns predictions from all models)
│   ├── inference.py            # Model inference script (predicts using all models)
├── tests/                      # Unit tests for pipeline
│   ├── test_pipeline.py        # Test script for preprocessing, inference, and API
├── don_prediction.ipynb        # Jupyter Notebook for Analysis and Model Development
├── streamlit_app.py            # Streamlit app for real-time predictions
├── generate_new_data.py        # Script to generate new data for inference
├── Dockerfile                  # Containerization instructions
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
```

---