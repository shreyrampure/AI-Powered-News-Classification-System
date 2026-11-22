
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# --------------------------------
# 1. LOAD DATASET
# --------------------------------

print("Loading dataset...")

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

print("Dataset loaded:", len(df))

# --------------------------------
# 2. TRAIN TEST SPLIT
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------
# 3. TF-IDF
# --------------------------------

print("\nApplying TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# --------------------------------
# 4. DEFINE MODELS
# --------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Naive Bayes": MultinomialNB(),

    "Linear SVM": LinearSVC(),

"Random Forest": RandomForestClassifier(
    n_estimators=30,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)
}

results = []

# --------------------------------
# 5. TRAIN AND EVALUATE
# --------------------------------

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# --------------------------------
# 6. SAVE RESULTS
# --------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nMODEL COMPARISON RESULTS")
print("------------------------")

print(results_df.to_string(index=False))

best_model = results_df.loc[
    results_df["Accuracy"].idxmax()
]

print("\nBEST MODEL:")
print(best_model)

print("\nComparison completed successfully!")
print("Results saved to model_comparison.csv")
