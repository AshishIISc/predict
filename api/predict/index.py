import pandas as pd
from fastapi import FastAPI
import joblib
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from pydantic import BaseModel

app = FastAPI()

# Load artifacts
xgb = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# model = DistilBertModel.from_pretrained("distilbert-base-uncased")
tokenizer = None
model = None


def load_model():
    from transformers import DistilBertTokenizer, DistilBertModel
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    return tokenizer, model


class Problem(BaseModel):
    worker_role: str                      # categorical
    worker_domain: str                    # categorical
    problem_title: str                    # (Used for text embeddings, not in structured features)
    problem_description: str              # (Used for text embeddings, not in structured features)
    type_of_issue: str                    # categorical
    public_reputation_risk: int           # Numeric (binary)
    impacted_users_estimate: int          # Numeric
    revenue_impact_estimate: float        # Numeric
    time_to_fix_estimate_hours: float     # Numeric


@app.get("/")
async def root():
    return {
        "message": "Worker Problem Classifier API",
        "endpoints": {
            "/predict": "POST problem data to get classification"
        }
    }


@app.post("/predict")
async def predict(problem: Problem):
    global tokenizer, model
    if tokenizer is None:
        tokenizer, model = load_model()
    # Text embeddings
    text = problem.problem_title + " " + problem.problem_description
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    text_embedding = outputs.last_hidden_state.mean(dim=1).numpy()

    # Structured features
    structured_data = pd.DataFrame(
        [{
            "impacted_users_estimate": problem.impacted_users_estimate,
            "revenue_impact_estimate": problem.revenue_impact_estimate,
            "time_to_fix_estimate_hours": problem.time_to_fix_estimate_hours,
            "public_reputation_risk": problem.public_reputation_risk,
            "worker_role": problem.worker_role,
            "worker_domain": problem.worker_domain,
            "type_of_issue": problem.type_of_issue
        }]
    )
    X_structured = preprocessor.transform(structured_data)

    # Combine and predict
    X = np.hstack([X_structured, text_embedding])
    pred = xgb.predict(X)[0]
    return {"prediction": "hair_on_fire" if pred == 1 else "vitamin"}
