
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

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

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. TF-IDF
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Train Linear SVM
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# 5. Predictions
predictions = model.predict(X_test_tfidf)

# 6. Evaluation metrics
accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")
macro_precision = precision_score(
    y_test, predictions, average="macro"
)
macro_recall = recall_score(
    y_test, predictions, average="macro"
)

print("\nMODEL EVALUATION")
print("-------------------------")

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 7. Save classification report
report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv")

# 8. Save overall metrics
metrics_df = pd.DataFrame([{
    "Model": "Linear SVM",
    "Accuracy": accuracy,
    "Macro Precision": macro_precision,
    "Macro Recall": macro_recall,
    "Macro F1": macro_f1
}])

metrics_df.to_csv("model_metrics.csv", index=False)

print("\nEvaluation completed!")
print("Classification report saved.")
print("Model metrics saved.")
