import pandas as pd
import joblib
import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizer, DistilBertModel

# Load test data (ensure no 'problem_id' column)
test_df = pd.read_csv("/Users/ashishkumar/Downloads/knowledge_worker_problems_test.csv")
test_df = test_df.drop(columns=['problem_id'], errors='ignore')  # Silently drop if missing

# Load trained artifacts
preprocessor = joblib.load("preprocessor.joblib")
xgb = joblib.load("model.joblib")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")


# Generate text embeddings (same as training)
def get_embeddings(texts):
    inputs = tokenizer(
        texts.tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


test_text_embeddings = get_embeddings(
    test_df["problem_title"] + " " + test_df["problem_description"]
)

# Preprocess structured features
X_test_structured = preprocessor.transform(test_df)  # Uses the same preprocessor
X_test = np.hstack([X_test_structured, test_text_embeddings])

# Prepare true labels
y_test = test_df["label"].map({"hair_on_fire": 1, "vitamin": 0})


# Get model predictions
y_pred = xgb.predict(X_test)

# Add predictions back to DataFrame for inspection
test_df["predicted_label"] = np.where(y_pred == 1, "hair_on_fire", "vitamin")
test_df["predicted_prob"] = xgb.predict_proba(X_test)[:, 1]  # Probability of "hair_on_fire"


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["vitamin", "hair_on_fire"]))

print("\nConfusion Matrix:")
print(pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]))

# Save test results with predictions
test_df.to_csv("test_results_with_predictions.csv", index=False)

# Save misclassified examples
misclassified = test_df[test_df["label"] != test_df["predicted_label"]]
misclassified.to_csv("misclassified_examples.csv", index=False)
