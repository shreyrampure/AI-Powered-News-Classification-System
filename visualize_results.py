
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# 1. LOAD MODEL COMPARISON RESULTS
# --------------------------------

df = pd.read_csv("model_comparison.csv")

print("\nMODEL COMPARISON DATA")
print(df)

sns.set_style("whitegrid")

# --------------------------------
# GRAPH 1: ALL METRICS COMPARISON
# --------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

df_melted = df.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(12, 7))

sns.barplot(
    data=df_melted,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Comparison of All Models - Performance Metrics")
plt.ylabel("Score")
plt.xlabel("Performance Metrics")
plt.ylim(0, 1)
plt.legend(title="Machine Learning Models")
plt.tight_layout()

plt.savefig("comparison_all_metrics.png", dpi=300)
plt.show()
plt.close()

# --------------------------------
# GRAPH 2: ACCURACY COMPARISON
# --------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Model",
    y="Accuracy"
)

plt.title("Accuracy Comparison of All Models")
plt.ylabel("Accuracy")
plt.xlabel("Machine Learning Models")
plt.ylim(0, 1)
plt.xticks(rotation=15)

for i, value in enumerate(df["Accuracy"]):
    plt.text(i, value + 0.01, f"{value:.2%}", ha="center")

plt.tight_layout()
plt.savefig("comparison_accuracy.png", dpi=300)
plt.show()
plt.close()

# --------------------------------
# GRAPH 3: F1 SCORE COMPARISON
# --------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Model",
    y="F1 Score"
)

plt.title("F1 Score Comparison of All Models")
plt.ylabel("F1 Score")
plt.xlabel("Machine Learning Models")
plt.ylim(0, 1)
plt.xticks(rotation=15)

for i, value in enumerate(df["F1 Score"]):
    plt.text(i, value + 0.01, f"{value:.2%}", ha="center")

plt.tight_layout()
plt.savefig("comparison_f1_score.png", dpi=300)
plt.show()
plt.close()

print("\nAll comparison graphs generated successfully!")
