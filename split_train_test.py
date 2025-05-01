import pandas as pd

# Load the original dataset
df = pd.read_csv("/Users/ashishkumar/Downloads/knowledge_worker_problems.csv")

# Verify class distribution
print("Original Class Distribution:")
print(df["label"].value_counts(normalize=True))

from sklearn.model_selection import train_test_split

# Split with stratification (preserves class ratios)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,                # 20% test, 80% train
    random_state=42,              # For reproducibility
    stratify=df["label"]          # Ensures imbalance is preserved
)

# Verify splits
print("\nTraining Set Class Distribution:")
print(train_df["label"].value_counts(normalize=True))

print("\nTest Set Class Distribution:")
print(test_df["label"].value_counts(normalize=True))


# Save test set to a new CSV
test_df.to_csv("/Users/ashishkumar/Downloads/knowledge_worker_problems_test.csv", index=False)

# Optionally save train set (if needed)
train_df.to_csv("/Users/ashishkumar/Downloads/knowledge_worker_problems_train.csv", index=False)

# Quick verification
test_check = pd.read_csv("/Users/ashishkumar/Downloads/knowledge_worker_problems_test.csv")
# Verify
print("Test Set Class Distribution:")
print(test_df["label"].value_counts(normalize=True))
print("\nTest Set Sample:")
print(test_check.head())
