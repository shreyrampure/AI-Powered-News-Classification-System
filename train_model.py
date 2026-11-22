
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

from imblearn.over_sampling import RandomOverSampler

# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

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

print("Dataset shape:", df.shape)
print("\nCategory distribution:")
print(y.value_counts())

# --------------------------------------------------
# 2. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 3. TF-IDF VECTORIZATION
# --------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)

# --------------------------------------------------
# 4. RANDOM OVERSAMPLING
# --------------------------------------------------

oversampler = RandomOverSampler(random_state=42)

X_train_over, y_train_over = oversampler.fit_resample(
    X_train_tfidf,
    y_train
)

print("\nBefore oversampling:")
print(y_train.value_counts())

print("\nAfter oversampling:")
print(y_train_over.value_counts())

# --------------------------------------------------
# 5. DEFINE EXPERIMENTS
# --------------------------------------------------

experiments = [
    {
        "name": "Logistic Regression - Original",
        "model": LogisticRegression(max_iter=1000),
        "X": X_train_tfidf,
        "y": y_train
    },
    {
        "name": "Logistic Regression - Oversampling",
        "model": LogisticRegression(max_iter=1000),
        "X": X_train_over,
        "y": y_train_over
    },
    {
        "name": "Logistic Regression - Class Weight",
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "X": X_train_tfidf,
        "y": y_train
    },
    {
        "name": "Logistic Regression - Oversampling + Class Weight",
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "X": X_train_over,
        "y": y_train_over
    },
    {
        "name": "Linear SVM - Original",
        "model": LinearSVC(),
        "X": X_train_tfidf,
        "y": y_train
    },
    {
        "name": "Linear SVM - Oversampling",
        "model": LinearSVC(),
        "X": X_train_over,
        "y": y_train_over
    },
    {
        "name": "Linear SVM - Class Weight",
        "model": LinearSVC(class_weight="balanced"),
        "X": X_train_tfidf,
        "y": y_train
    },
    {
        "name": "Linear SVM - Oversampling + Class Weight",
        "model": LinearSVC(class_weight="balanced"),
        "X": X_train_over,
        "y": y_train_over
    }
]

# --------------------------------------------------
# 6. TRAIN AND EVALUATE
# --------------------------------------------------

results = []

for experiment in experiments:

    print("\n" + "=" * 65)
    print("MODEL:", experiment["name"])
    print("=" * 65)

    model = experiment["model"]

    model.fit(
        experiment["X"],
        experiment["y"]
    )

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    results.append({
        "Model": experiment["name"],
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1
    })

# --------------------------------------------------
# 7. COMPARE ALL EXPERIMENTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to model_comparison.csv")
