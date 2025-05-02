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

### 2. Install requirements
```bash
pip install -r requirements.txt
```

## Train the model
```bash
python train.py
```
This will:

Process the training data (knowledge_worker_problems_train.csv)

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

## Requirements.txt
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

