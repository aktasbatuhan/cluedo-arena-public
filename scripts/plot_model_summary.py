import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the summary scores CSV
df = pd.read_csv("/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/results/evaluation_reports/all_providers_default_eval/comprehensive_evaluation_report_20250511_162229_summary_scores.csv")

# Set up the plot style
sns.set(style="whitegrid")
metrics = ["Average_Precision", "Average_Recall", "Average_F1_Score", "Average_ExactMatch"]

# Melt the DataFrame for easier plotting
df_melted = df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 7))
sns.barplot(
    data=df_melted,
    x="Score",
    y="Model",
    hue="Metric",
    palette="viridis"
)
plt.title("Model Performance Comparison")
plt.xlabel("Score")
plt.ylabel("Model")
plt.legend(title="Metric")
plt.tight_layout()
plt.show()