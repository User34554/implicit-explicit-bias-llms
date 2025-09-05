import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, f1_score

# --- Step 1: Load your Excel file ---
input_file = "results.xlsx"  # Replace with your Excel file path
output_file = "metrics_results.xlsx"

df = pd.read_excel(input_file, sheet_name="third_dataset")

# --- Step 2: Define models and ground truth ---
ground_truth_col = "ground_truth human"  # Replace with your ground truth column name
model_cols = ["trained", "untrained", "Wizard"]  # Replace with your LLM column names

# --- Step 3: Compute metrics ---
results = []

for model in model_cols:
    precision = precision_score(df[ground_truth_col], df[model])
    accuracy = accuracy_score(df[ground_truth_col], df[model])
    f1 = f1_score(df[ground_truth_col], df[model])

    results.append({
        "Model": model,
        "Precision": precision,
        "Accuracy": accuracy,
        "F1": f1
    })

# --- Step 4: Convert to DataFrame ---
metrics_df = pd.DataFrame(results)

# --- Step 5: Save to Excel ---
metrics_df.to_excel(output_file, index=False)

print(f"Metrics saved to {output_file}")
print(metrics_df)