import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
# from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()

# Initialize models (cold start optimization)
xgb_session = None
text_model = None
preprocessor_params = None


def load_models():
    global xgb_session, text_model, preprocessor_params

    if xgb_session is None:
        xgb_session = ort.InferenceSession(os.path.join(os.path.dirname(__file__), "model.onnx"))

    # if text_model is None:
        # text_model = SentenceTransformer('all-MiniLM-L6-v2')

    if preprocessor_params is None:
        preprocessor_params = np.load(
            os.path.join(os.path.dirname(__file__), "preprocessor_params.npz"),
            allow_pickle=True
        )

    # output details
    output_details = xgb_session.get_outputs()
    print("Model outputs:", [output.name for output in output_details])
    print("Output shapes:", [xgb_session.get_outputs()[0].shape])


API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}


def get_embeddings(texts):
    response = requests.post(API_URL, headers=headers, json={"inputs": texts})
    return response.json()


class Problem(BaseModel):
    worker_role: str  # categorical
    worker_domain: str  # categorical
    problem_title: str  # (Used for text embeddings, not in structured features)
    problem_description: str  # (Used for text embeddings, not in structured features)
    type_of_issue: str  # categorical
    public_reputation_risk: int  # Numeric (binary)
    impacted_users_estimate: int  # Numeric
    revenue_impact_estimate: float  # Numeric
    time_to_fix_estimate_hours: float  # Numeric


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
    load_models()

    # 1. Text embeddings (384D)
    text = problem.problem_title + " " + problem.problem_description
    # text_embedding = text_model.encode([text])[0][:384]  # Ensure 384 dimensions
    text_embedding = get_embeddings([text])[0][:384]

    # 2. Numeric features (4)
    numeric_data = (np.array([
        problem.impacted_users_estimate,
        problem.revenue_impact_estimate,
        problem.time_to_fix_estimate_hours,
        problem.public_reputation_risk
    ], dtype=np.float32) - preprocessor_params['num_mean']) / preprocessor_params['num_scale']

    # 3. Categorical features (412 - 384 - 4 = 24)
    cat_data = np.zeros(24)  # Update this number!
    if problem.type_of_issue in preprocessor_params['cat_categories'][0]:
        idx = list(preprocessor_params['cat_categories'][0]).index(problem.type_of_issue)
        cat_data[idx] = 1

    # 4. Combine all features
    X = np.concatenate([numeric_data, cat_data, text_embedding]).astype(np.float32).reshape(1, -1)

    # 5. Verify
    assert X.shape[1] == 412, f"Expected 412 features, got {X.shape[1]}"

    # 6. Predict
    outputs = xgb_session.run(None, {'float_input': X})

    # Handle different output formats
    if isinstance(outputs[0], np.ndarray):
        # Standard array output
        pred_score = outputs[0].item()  # Gets the scalar value
    else:
        # Raw scalar output
        pred_score = float(outputs[0])

    print("outputs", outputs)

    return {"prediction": "hair_on_fire" if pred_score > 0.5 else "vitamin"}

