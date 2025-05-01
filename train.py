import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_csv("/Users/ashishkumar/Downloads/knowledge_worker_problems_train.csv")
df = df.drop(columns=['problem_id'])

# Text embeddings (DistilBERT)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")


def get_embeddings(texts):
    text_list = texts.tolist() if isinstance(texts, pd.Series) else texts      # added this becuase of ValueError: text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
    inputs = tokenizer(
        text_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128          # Shorter max_length for speed
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


text_embeddings = get_embeddings(df["problem_title"] + " " + df["problem_description"])

# Structured features
numeric_features = [
    "impacted_users_estimate",
    "revenue_impact_estimate",
    "time_to_fix_estimate_hours",
    "public_reputation_risk"
]
categorical_features = [
    "worker_role",
    "worker_domain",
    "type_of_issue"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    sparse_threshold=0  # All outputs will be dense       # added this because of ValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has 2 dimension(s)
)

X_structured = preprocessor.fit_transform(df)
print("Feature names after preprocessing:", preprocessor.get_feature_names_out())
X = np.hstack([X_structured, text_embeddings])
y = df["label"].map({"hair_on_fire": 1, "vitamin": 0})

print("X shape:", X.shape)  # Should be (n_samples, n_features)
print("y shape:", y.shape)  # Should be (n_samples,)

# Train XGBoost (optimized for small data)
xgb = XGBClassifier(
    n_estimators=100,  # Fewer trees for small data
    max_depth=3,       # Prevent overfitting
    learning_rate=0.1
)
xgb.fit(X, y)

# Save artifacts
joblib.dump(xgb, "model.joblib")
joblib.dump(preprocessor, "preprocessor.joblib")
