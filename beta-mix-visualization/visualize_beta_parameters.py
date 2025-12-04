import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_DIR = os.path.join(REPO_DIR, "finetuning")

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ---------------------------------------------------------
# Load .jsonl file
# ---------------------------------------------------------
def load_data(file_path):
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Plot 1: Parameter traces over time
# ---------------------------------------------------------
def plot_parameters_over_time(df):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    params = ["alpha1", "beta1", "alpha2", "beta2", "mix_prob"]

    for i, p in enumerate(params):
        ax = axes[i // 2][i % 2]
        ax.plot(df["step"], df[p])
        ax.set_title(f"{p} over time")
        ax.set_xlabel("Step")
        ax.set_ylabel(p)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("parameters_over_time.png")
    plt.show()


# ---------------------------------------------------------
# Plot 2: Mean + std of sampled difficulties over time
# ---------------------------------------------------------
def plot_difficulty_summary(df):
    df["mean_difficulty"] = df["sampled_difficulties"].apply(lambda arr: np.mean(arr))
    df["std_difficulty"] = df["sampled_difficulties"].apply(lambda arr: np.std(arr))

    plt.figure(figsize=(12, 6))
    plt.plot(df["step"], df["mean_difficulty"], label="Mean Difficulty")
    plt.fill_between(
        df["step"],
        df["mean_difficulty"] - df["std_difficulty"],
        df["mean_difficulty"] + df["std_difficulty"],
        alpha=0.2,
        label="±1 std",
    )

    plt.title("Sampled Difficulty Mean ± Std Over Time")
    plt.xlabel("Step")
    plt.ylabel("Difficulty")
    plt.grid(True)
    plt.legend()
    plt.savefig("difficulty_mean_std_over_time.png")
    plt.show()


# ---------------------------------------------------------
# Plot 3: Rolling-window histogram of difficulties
# Meaningful because each window has many samples.
# ---------------------------------------------------------
def plot_difficulty_rolling_hist(df, window=50):
    all_diffs = np.concatenate(df["sampled_difficulties"].values)
    steps = df["step"].values

    # Prepare for rolling windows
    window_samples = []
    window_centers = []

    flattened = []
    for step, diffs in zip(df["step"], df["sampled_difficulties"]):
        for d in diffs:
            flattened.append((step, d))

    flat_df = pd.DataFrame(flattened, columns=["step", "difficulty"])

    # Rolling windows by step
    plt.figure(figsize=(12, 6))

    for start in range(0, df["step"].max(), window):
        subset = flat_df[
            (flat_df["step"] >= start) & (flat_df["step"] < start + window)
        ]
        if len(subset) > 0:
            plt.hist(
                subset["difficulty"],
                bins=20,
                alpha=0.3,
                label=f"Steps {start}-{start+window}",
            )

    plt.title("Difficulty Distribution in Rolling Windows")
    plt.xlabel("Difficulty")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("rolling_difficulty_histograms.png")
    plt.show()


# ---------------------------------------------------------
# Plot 4: Heatmap of difficulty samples over time
# ---------------------------------------------------------
def plot_difficulty_heatmap(df):
    # Flatten difficulties into (step x sample_index)
    matrix = np.vstack(df["sampled_difficulties"].values)

    plt.figure(figsize=(14, 6))
    sns.heatmap(matrix, cmap="viridis", cbar=True)
    plt.title("Heatmap of Sampled Difficulties Over Time")
    plt.xlabel("Sample index within step")
    plt.ylabel("Training step")
    plt.savefig("difficulty_heatmap.png")
    plt.show()


# ---------------------------------------------------------
# Plot 5: Overall difficulty distribution
# ---------------------------------------------------------
def plot_overall_difficulty_distribution(df):
    all_difficulties = np.concatenate(df["sampled_difficulties"].values)

    plt.figure(figsize=(10, 6))
    plt.hist(all_difficulties, bins=25, alpha=0.7)
    plt.title("Overall Distribution of Sampled Difficulties")
    plt.xlabel("Difficulty")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("overall_difficulty_distribution.png")
    plt.show()


# ---------------------------------------------------------
# Plot 6: Parameter correlation matrix
# ---------------------------------------------------------
def plot_parameter_correlations(df):
    corr = df[
        [
            "alpha1",
            "beta1",
            "alpha2",
            "beta2",
            "mix_prob",
            "total_loss",
            "curriculum_loss",
            "avg_reward",
        ]
    ].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    plt.show()


# ---------------------------------------------------------
# Main generator
# ---------------------------------------------------------
def generate_and_save_plots(file_path):
    df = load_data(file_path)

    # Create output folder
    os.makedirs("plots", exist_ok=True)
    os.chdir("plots")

    plot_parameters_over_time(df)
    plot_difficulty_summary(df)
    plot_difficulty_rolling_hist(df, window=50)
    plot_difficulty_heatmap(df)
    plot_overall_difficulty_distribution(df)
    plot_parameter_correlations(df)

    print("All plots saved in ./plots")


if __name__ == "__main__":
    file_path = os.path.join(
        FINETUNE_DIR,
        "qwen_25_05b_bmc_fixed_final_v2",
        "logs",
        "active_training_log.jsonl",
    )
    generate_and_save_plots(file_path)
