import pandas as pd

# Load dataset
df = pd.read_csv("data/anxiety_dataset.csv")

print("Original Dataset:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Label mapping
label_map = {
    "low": 0,
    "moderate": 1,
    "high": 2
}

# Convert labels to numbers
df["label_num"] = df["label"].map(label_map)

print("\nDataset after label mapping:")
print(df.head())

# Save processed dataset
df.to_csv("data/processed_anxiety_dataset.csv", index=False)

print("\nProcessed dataset saved successfully!")