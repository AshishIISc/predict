# train_fixed.py
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import joblib

def get_embeddings(texts, model):
    """Process in chunks to avoid memory overload"""
    CHUNK_SIZE = 50
    embeddings = []
    for i in range(0, len(texts), CHUNK_SIZE):
        chunk = texts[i:i+CHUNK_SIZE]
        embeddings.append(model.encode(chunk))
        gc.collect()
    return np.vstack(embeddings)

print("Loading data...")
df = pd.read_csv("knowledge_worker_problems_train.csv")
df = df.drop(columns=['problem_id'])

print("Loading text model...")
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Generating embeddings...")
text_embeddings = get_embeddings(
    (df["problem_title"] + " " + df["problem_description"]).tolist(),
    text_model
)

print("Preprocessing...")
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["impacted_users_estimate", "revenue_impact_estimate", 
     "time_to_fix_estimate_hours", "public_reputation_risk"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["worker_role", "worker_domain", "type_of_issue"])
], sparse_threshold=0)

X = np.hstack([
    preprocessor.fit_transform(df),
    text_embeddings
])
y = df["label"].map({"hair_on_fire": 1, "vitamin": 0})

print("Training...")
xgb = XGBClassifier(
    n_estimators=50,  # Reduced for testing
    max_depth=3,
    learning_rate=0.1,
    tree_method='hist'  # More memory efficient
)
xgb.fit(X, y)
print("X shape:", X.shape)  # Should be (n_samples, n_features)
print(" y shape:", y.shape)  # Should be (n_samples,)
print("Feature names after preprocessing:", preprocessor.get_feature_names_out())


print("Saving...")
joblib.dump(xgb, "model.joblib")
joblib.dump(preprocessor, "preprocessor.joblib")

print("Training Done!")

# Save preprocessor components separately for ONNX
np.savez("preprocessor_params.npz",
         num_mean=preprocessor.transformers_[0][1].mean_,
         num_scale=preprocessor.transformers_[0][1].scale_,
         cat_categories=preprocessor.transformers_[1][1].categories_)

print("Preprocessor parameters saved.")

# Add this to your training script after preprocessing
print("Feature breakdown:")
print(f"- Categorical: {sum(len(cats) for cats in preprocessor.transformers_[1][1].categories_)}")
print(f"- Text: {text_embeddings.shape[1]}")  # Should be 384
print(f"Total: {X.shape[1]}")  # Should match 412
