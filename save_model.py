
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

df = df[["text_for_tfidf", "category"]].dropna()

df["text_for_tfidf"] = df["text_for_tfidf"].astype(str).str.strip()
df["category"] = df["category"].astype(str).str.strip()

df = df[
    (df["text_for_tfidf"] != "") &
    (df["category"] != "")
]

X = df["text_for_tfidf"]
y = df["category"]

# 2. Convert text into TF-IDF features
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(X)

# 3. Train Linear SVM on the complete dataset
model = LinearSVC()
model.fit(X_tfidf, y)

# 4. Create model folder
os.makedirs("model", exist_ok=True)

# 5. Save model and vectorizer
joblib.dump(model, "model/news_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel trained and saved successfully!")
print("Saved: model/news_model.pkl")
print("Saved: model/tfidf_vectorizer.pkl")
