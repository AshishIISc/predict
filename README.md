# Worker Problem Classifier API

A FastAPI-based service that classifies worker problems using **DistilBert** and **XGBoost**.

## Prerequisites

- Python 3.9+
- pip package manager

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
python -m venv venv_predict
source venv_predict/bin/activate
```

### 2. Install requirements  (for training purpose, and this would be root directory requirements.txt)
```bash
pip install -r requirements.txt
```

## Train the model
### 1. create and Save preprocessor components separately for ONNX
```bash
python train_with_onxx.py
```
This will:

Process the training data (knowledge_worker_problems_train.csv) and will train the model and save it as model.joblib and alos make preprocessor_params.npz

### 2. convert the model.joblib to model.onnx (doing this so that while inference we will directly use the onnx model instead of loading xgboost which will take lot of memory)
```bash
python convert_to_onnx.py
```
this file will get created in root directory, copy this model.onnx to api/predict/
```bash
cp model.onnx api/predict/model.onnx
```
also copy preprocessor_params.npz from root to api/predict/
```bash
cp preprocessor_params.npz api/predict/preprocessor_params.npz
```
### 3. create a requirement file in api/predict/ folder   (api/predict/requirements.txt)
```plaintext
onnxruntime==1.17.0
requests==2.32.3
joblib==1.3.2
numpy==1.26.0
pydantic==2.11.4
fastapi==0.115.12
```

## Run the api server
```bash
cd ./api/predict
<path_till_venv_predict>/bin/python -m uvicorn index:app
```

## Test the API
```python
import requests

data = {
    "worker_role": "developer",
    "worker_domain": "backend",
    "problem_title": "Database connection issues",
    "problem_description": "Unable to connect to production database",
    "type_of_issue": "technical",
    "public_reputation_risk": 1,
    "impacted_users_estimate": 500,
    "revenue_impact_estimate": 25000.0,
    "time_to_fix_estimate_hours": 8.0
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## API test Bash
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "worker_role": "developer",
    "worker_domain": "backend",
    "problem_title": "Database connection issues",
    "problem_description": "Unable to connect to production database",
    "type_of_issue": "technical",
    "public_reputation_risk": 1,
    "impacted_users_estimate": 500,
    "revenue_impact_estimate": 25000.0,
    "time_to_fix_estimate_hours": 8.0
}'
```

## File Structure
```plaintext
predict/
├── train.py
├── knowledge_worker_problems_train.csv
├── index.py
├── requirements.txt
├── vercel.json
├── model.joblib            (generated after training)
└── preprocessor.joblib     (generated after training)
```

## Requirements.txt (these are for training purpose, this file should be in root directory)
```plaintext
fastapi==0.109.2
uvicorn==0.27.0
scikit-learn==1.6.1
joblib==1.3.2
scikit-learn>=1.0.0
xgboost>=1.5.0
transformers>=4.12.0
torch>=1.9.0
```

