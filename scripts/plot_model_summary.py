import glob
import argparse
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(summary_scores_path: Optional[str] = None) -> None:
    """Plot model performance summary scores.

    Parameters
    ----------
    summary_scores_path: optional str
        Path to a CSV file containing summary scores. If not provided the
        function searches ``results/evaluation_reports`` for a matching file.
    """
    if summary_scores_path is None:
        matches = glob.glob(
            "results/evaluation_reports/**/*_summary_scores.csv", recursive=True
        )
        if not matches:
            raise FileNotFoundError(
                "No summary scores CSV found in results/evaluation_reports"
            )
        summary_scores_path = matches[0]

    # Load the summary scores CSV
    df = pd.read_csv(summary_scores_path)

    # Set up the plot style
    sns.set(style="whitegrid")
    metrics = [
        "Average_Precision",
        "Average_Recall",
        "Average_F1_Score",
        "Average_ExactMatch",
    ]

    # Melt the DataFrame for easier plotting
    df_melted = df.melt(
        id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score"
    )

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_melted,
        x="Score",
        y="Model",
        hue="Metric",
        palette="viridis",
    )
    plt.title("Model Performance Comparison")
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot summary scores for model evaluation"
    )
    parser.add_argument(
        "--summary-scores-path",
        default=None,
        help="Path to the summary_scores.csv file. If not provided, searches in results/evaluation_reports.",
    )
    args = parser.parse_args()
    main(args.summary_scores_path)

