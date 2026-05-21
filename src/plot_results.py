import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("results/tables/main_results.csv")

supervised_df = df[df["method"] == "supervised_finetune"]
ssl_df = df[df["method"] == "ssl_fixmatch"]

if len(ssl_df) > 0:
    best_threshold = ssl_df.groupby("threshold")["test_accuracy"].mean().idxmax()
    ssl_plot_df = ssl_df[ssl_df["threshold"] == best_threshold]
else:
    ssl_plot_df = ssl_df

supervised_df = supervised_df.sort_values("label_fraction")
ssl_plot_df = ssl_plot_df.sort_values("label_fraction")

plt.figure(figsize=(8, 5))

if len(supervised_df) > 0:
    plt.plot(
        supervised_df["label_fraction"],
        supervised_df["test_accuracy"],
        marker="o",
        label="Supervised Fine-Tuning"
    )

if len(ssl_plot_df) > 0:
    plt.plot(
        ssl_plot_df["label_fraction"],
        ssl_plot_df["test_accuracy"],
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

plt.figure(figsize=(8, 5))

if len(supervised_df) > 0:
    plt.plot(
        supervised_df["label_fraction"],
        supervised_df["macro_f1"],
        marker="o",
        label="Supervised Fine-Tuning"
    )

if len(ssl_plot_df) > 0:
    plt.plot(
        ssl_plot_df["label_fraction"],
        ssl_plot_df["macro_f1"],
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

ablation_df = df[
    (df["method"] == "ssl_fixmatch") &
    (df["label_fraction"] == 0.1)
].sort_values("threshold")

if len(ablation_df) > 0:
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