import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/tables/main_results.csv")

# Separate supervised and SSL
supervised_df = df[df["method"] == "supervised_finetune"]
ssl_df = df[
    (df["method"] == "ssl_fixmatch") &
    (df["threshold"] == 0.5)
]

# Sort for plotting
supervised_df = supervised_df.sort_values("label_fraction")
ssl_df = ssl_df.sort_values("label_fraction")

# Plot accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    supervised_df["label_fraction"],
    supervised_df["test_accuracy"],
    marker="o",
    label="Supervised Fine-Tuning"
)

plt.plot(
    ssl_df["label_fraction"],
    ssl_df["test_accuracy"],
    marker="o",
    label="SSL FixMatch"
)

plt.xscale("log")

plt.xlabel("Label Fraction")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Label Fraction")
plt.legend()

plt.savefig("results/figures/accuracy_vs_labels.png")
plt.close()

# Plot macro F1
plt.figure(figsize=(8, 5))

plt.plot(
    supervised_df["label_fraction"],
    supervised_df["macro_f1"],
    marker="o",
    label="Supervised Fine-Tuning"
)

plt.plot(
    ssl_df["label_fraction"],
    ssl_df["macro_f1"],
    marker="o",
    label="SSL FixMatch"
)

plt.xscale("log")

plt.xlabel("Label Fraction")
plt.ylabel("Macro F1")
plt.title("Macro F1 vs Label Fraction")
plt.legend()

plt.savefig("results/figures/f1_vs_labels.png")
plt.close()

# Threshold ablation plot
ablation_df = df[
    (df["method"] == "ssl_fixmatch") &
    (df["label_fraction"] == 0.1)
]

plt.figure(figsize=(8, 5))

plt.plot(
    ablation_df["threshold"],
    ablation_df["test_accuracy"],
    marker="o"
)

plt.xlabel("Confidence Threshold")
plt.ylabel("Test Accuracy")
plt.title("SSL Threshold Ablation (10% Labels)")

plt.savefig("results/figures/threshold_ablation.png")
plt.close()

print("Saved all plots.")