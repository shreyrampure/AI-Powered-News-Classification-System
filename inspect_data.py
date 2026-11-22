import pandas as pd

df = pd.read_csv("data/dataset.csv")

print("DATASET SHAPE:", df.shape)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nFIRST 5 RECORDS:")
print(df.head())

print("\nCATEGORY DISTRIBUTION:")
print(df["category"].value_counts())

print("\nNUMBER OF CATEGORIES:")
print(df["category"].nunique())