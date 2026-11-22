
import pandas as pd

# Load dataset
df = pd.read_json("data/dataset.json", lines=True)

print("Original dataset:", df.shape)

# Combine headline and description
df["text"] = (
    df["headline"].fillna("") + " " +
    df["short_description"].fillna("")
)

# Keep only required columns
df = df[["text", "category"]]

# Remove empty text
df["text"] = df["text"].str.strip()
df = df[df["text"] != ""]

# Remove duplicate records
df = df.drop_duplicates(subset=["text"])

# Reset index
df = df.reset_index(drop=True)

print("\nPREPROCESSING COMPLETED")
print("Total articles:", len(df))
print("Total categories:", df["category"].nunique())

print("\nCategory distribution:")
print(df["category"].value_counts())

# Save cleaned dataset
df.to_csv("data/cleaned_news.csv", index=False)

print("\nCleaned dataset saved successfully!")
