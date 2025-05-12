#!/usr/bin/env python3
import os
import pandas as pd
import json
import yaml # For parsing Cohere responses
import ast  # For safely evaluating string representations of lists
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# --- Configuration ---
# File paths (can be overridden by command-line arguments)
INPUT_CSV_REPORT_PATH = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/results/evaluation_reports/comprehensive_evaluation_report_20250510_160744.csv"
INPUT_JSON_REPORT_PATH = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/results/evaluation_reports/comprehensive_evaluation_report_20250510_160744.json"
OUTPUT_DETAILED_CSV_PATH = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/results/evaluation_reports/evaluation_report_detailed.csv"
OUTPUT_SUMMARY_CSV_PATH = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/results/evaluation_reports/evaluation_summary_scores.csv"
OUTPUT_CHART_DIR = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/results/evaluation_charts"

# --- Helper Functions for Scoring ---

def safe_literal_eval(val):
    """Safely evaluate a string representation of a Python literal (e.g., a list)."""
    if pd.isna(val) or val == "[]" or val == "": # Handle NaN, empty list string, or empty string
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError, TypeError):
        # If it's not a list string, but a simple string, try splitting by common delimiters
        if isinstance(val, str):
            # Example: "Card1, Card2" or "Card1; Card2"
            # This is a basic attempt; might need refinement based on actual string format
            for delimiter in [", ", ",", "; ", ";"]:
                if delimiter in val:
                    return [item.strip() for item in val.split(delimiter)]
            return [val.strip()] # Treat as a single item list if no common delimiter
        return [] # Return empty list if not parsable

def parse_cohere_response_yaml(yaml_string):
    """Parse YAML string from Cohere response to get newlyDeducedCards."""
    if pd.isna(yaml_string) or not isinstance(yaml_string, str) or yaml_string.strip() == "":
        return []
    
    cleaned_yaml_string = yaml_string.strip()
    # Remove common Markdown code block fences
    if cleaned_yaml_string.startswith("```yaml") and cleaned_yaml_string.endswith("```"):
        cleaned_yaml_string = cleaned_yaml_string[len("```yaml"):-(len("```"))].strip()
    elif cleaned_yaml_string.startswith("```") and cleaned_yaml_string.endswith("```"):
        cleaned_yaml_string = cleaned_yaml_string[len("```"):-(len("```"))].strip()
        
    try:
        data = yaml.safe_load(cleaned_yaml_string)
        if isinstance(data, dict) and "newlyDeducedCards" in data:
            deductions = data["newlyDeducedCards"]
            if deductions is None: return [] # Handle explicit null
            if isinstance(deductions, list):
                return [str(item) for item in deductions if item is not None]
            elif isinstance(deductions, str):
                return [deductions.strip()]
        # Fallback: sometimes the response might be *just* the list, not a dict
        elif isinstance(data, list):
            return [str(item) for item in data if item is not None]
        return []
    except yaml.YAMLError as e:
        # print(f"  Warning: Could not parse YAML for response: {e}\nCleaned string was: {cleaned_yaml_string[:200]}...")
        return []
    except Exception as e:
        # print(f"  Unexpected error parsing response: {e}\nCleaned string was: {cleaned_yaml_string[:200]}...")
        return []

def calculate_metrics(predicted_list, ground_truth_list):
    """Calculate Precision, Recall, F1, and Exact Match for lists of items."""
    if not isinstance(predicted_list, list): predicted_list = []
    if not isinstance(ground_truth_list, list): ground_truth_list = []

    # Normalize: Convert to set of unique, lowercased, stripped strings
    # This handles minor variations like case, duplicates, and leading/trailing spaces
    pred_set = {str(p).strip().lower() for p in predicted_list if p}
    gt_set = {str(gt).strip().lower() for gt in ground_truth_list if gt}

    if not gt_set and not pred_set: # Both empty, perfect match for emptiness
        return 1.0, 1.0, 1.0, 1.0 
    if not gt_set and pred_set: # Ground truth empty, but prediction not
        return 0.0, 1.0, 0.0, 0.0 # Precision 0, Recall 1 (no false negatives, but false positives)
    if gt_set and not pred_set: # Prediction empty, but ground truth not
        return 1.0, 0.0, 0.0, 0.0 # Precision 1 (no false positives, but false negatives)

    true_positives = len(pred_set.intersection(gt_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0.0 # Should be safe due to earlier checks
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred_set == gt_set else 0.0
    
    return precision, recall, f1, exact_match

# --- Main Processing Logic ---
def main(args):
    print(f"Loading evaluation report from: {args.csv_report}")
    try:
        df_full = pd.read_csv(args.csv_report)
    except FileNotFoundError:
        print(f"Error: CSV report file not found at {args.csv_report}")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- 1. Separate original data and Cohere data ---
    df_cohere_results = df_full[df_full["model_provider"] == "Cohere"].copy()
    # Assuming original data doesn't have "model_provider" or it's different
    # A more robust way might be to identify rows that *don't* have Cohere model names
    # For now, this relies on Cohere results having the model_provider field set.
    original_data_indices = df_full[df_full["model_provider"] != "Cohere"].index
    if original_data_indices.empty and not df_cohere_results.empty:
        # If all rows are cohere results, it implies the original CSV only had prompts
        # or we need a different way to get original prompts and ground truths.
        # This simple split assumes the CSV had original results first, then Cohere results appended.
        
        # Let's try to find rows that have the 'Prompt' column but are NOT Cohere results
        # (This relies on `run_cohere_evaluation.py` using `prompt_text` for cohere rows)
        df_original_prompts_gt = df_full[df_full['Prompt'].notna() & (df_full['model_provider'] != 'Cohere')]
        if df_original_prompts_gt.empty:
             # Fallback: if all rows are Cohere, try to get unique prompts from Cohere data
             # and assume GT is in the *first* instance of that prompt in the original df_full.
            print("Warning: Could not definitively separate original data. Trying to map GT from full CSV.")
            prompt_to_gt_map = {}
            # Iterate through unique prompts seen in cohere results
            unique_prompts_in_cohere = df_cohere_results['prompt_text'].unique()
            for prompt_text_val in unique_prompts_in_cohere:
                # Find first original row with this prompt to get GT
                original_row_for_prompt = df_full[
                    (df_full['Prompt'] == prompt_text_val) &
                    (df_full['Ground Truth Deductions'].notna())
                ].iloc[0] if not df_full[
                    (df_full['Prompt'] == prompt_text_val) &
                    (df_full['Ground Truth Deductions'].notna())
                ].empty else None
                
                if original_row_for_prompt is not None:
                    prompt_to_gt_map[prompt_text_val] = safe_literal_eval(original_row_for_prompt['Ground Truth Deductions'])
                else:
                    prompt_to_gt_map[prompt_text_val] = [] # Default to empty if no GT found
            print(f"Created GT map for {len(prompt_to_gt_map)} prompts.")
        else:
            prompt_to_gt_map = pd.Series(df_original_prompts_gt['Ground Truth Deductions'].apply(safe_literal_eval).values, index=df_original_prompts_gt['Prompt']).to_dict()
            print(f"Created GT map from original data for {len(prompt_to_gt_map)} prompts.")
    else:
        # df_original_prompts_gt = df_full.loc[original_data_indices, ['Prompt', 'Ground Truth Deductions']].drop_duplicates(subset=['Prompt'])
        df_original_prompts_gt = df_full[df_full['Prompt'].notna() & df_full['Ground Truth Deductions'].notna()].drop_duplicates(subset=['Prompt'])
        prompt_to_gt_map = pd.Series(df_original_prompts_gt['Ground Truth Deductions'].apply(safe_literal_eval).values, index=df_original_prompts_gt['Prompt']).to_dict()
        print(f"Created GT map from original data for {len(prompt_to_gt_map)} prompts.")

    if not prompt_to_gt_map:
        print("Error: Could not create a map from Prompts to Ground Truth Deductions. Check your CSV structure.")
        print("Ensure 'Prompt' and 'Ground Truth Deductions' columns exist and are correctly populated in the original data.")
        return

    # --- 2. Process Cohere results ---
    print(f"Processing {len(df_cohere_results)} Cohere API results...")
    cohere_processed_data = []
    
    # Add a counter for debugging prints
    debug_print_count = 0
    MAX_DEBUG_PRINTS = 3 # Print for the first few Cohere responses

    for index, row in df_cohere_results.iterrows():
        prompt_text = row["prompt_text"]
        model_name = row["model_name"]
        generated_response = row["generated_response"]
        
        ground_truth = prompt_to_gt_map.get(prompt_text, []) # Get GT for this prompt

        if debug_print_count < MAX_DEBUG_PRINTS:
            print(f"\n--- Debugging Cohere Response (Entry {debug_print_count+1}) ---")
            print(f"Model: Cohere_{model_name}")
            print(f"Prompt: {prompt_text[:150]}...")
            print(f"Raw Generated Response:\n{generated_response}")
        
        parsed_deductions = parse_cohere_response_yaml(generated_response)
        
        if debug_print_count < MAX_DEBUG_PRINTS:
            print(f"Parsed Deductions: {parsed_deductions}")
            print(f"Ground Truth: {ground_truth}")
            print("---------------------------------------------")
            debug_print_count += 1
            
        precision, recall, f1, exact_match = calculate_metrics(parsed_deductions, ground_truth)
        
        cohere_processed_data.append({
            "prompt_text": prompt_text,
            "model_name": f"Cohere_{model_name}", # Prefix to distinguish from other models
            "parsed_deductions": parsed_deductions,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ExactMatch": exact_match
        })
    
    df_cohere_scored = pd.DataFrame(cohere_processed_data)

    # --- 3. Consolidate Data for Detailed CSV ---
    # Start with unique prompts and their ground truths
    df_detailed = pd.DataFrame(list(prompt_to_gt_map.items()), columns=['Prompt', 'Ground_Truth_Deductions'])

    # Merge original model scores (if they exist and have a common structure)
    # This is a simplified merge. If original data has scores, we need to identify those columns.
    # Assuming original_df had scores per prompt, we might need to pivot it first or select relevant columns.
    # For now, let's just keep the GT and add Cohere model scores.
    # If you have pre-calculated scores for other models in `df_full` (e.g. 'Command-A F1'),
    # they would need to be merged here based on 'Prompt'.

    # Pivot Cohere scored data: one row per prompt, columns for each model_metric
    if not df_cohere_scored.empty:
        # Define aggregation functions for each value type
        # For 'parsed_deductions' (list of strings), we take the 'first' encountered (should be unique per group)
        # For numeric scores, 'mean' is fine (should also be unique per group here)
        agg_functions = {
            'parsed_deductions': 'first',
            'Precision': 'mean',
            'Recall': 'mean',
            'F1-score': 'mean',
            'ExactMatch': 'mean'
        }
        
        df_cohere_pivot = df_cohere_scored.pivot_table(
            index='prompt_text',
            columns='model_name',
            values=['parsed_deductions', 'Precision', 'Recall', 'F1-score', 'ExactMatch'],
            aggfunc=agg_functions
        )
        # Flatten multi-index columns (e.g., ('F1-score', 'Cohere_command-r-plus'))
        df_cohere_pivot.columns = [f'{col[1]}_{col[0].replace("-score", "")}' for col in df_cohere_pivot.columns.values]
        df_cohere_pivot.reset_index(inplace=True)
        
        # Merge with the detailed DataFrame
        df_detailed = pd.merge(df_detailed, df_cohere_pivot, left_on='Prompt', right_on='prompt_text', how='left')
        if 'prompt_text' in df_detailed.columns: # Drop redundant prompt_text column after merge
            df_detailed.drop(columns=['prompt_text'], inplace=True)

    print(f"Saving detailed evaluation results to: {args.output_detailed_csv}")
    df_detailed.to_csv(args.output_detailed_csv, index=False)

    # --- 4. Aggregate Scores for Summary and Visualization ---
    # Identify all F1 score columns to average them
    f1_columns = [col for col in df_detailed.columns if 'F1' in col and 'Cohere' in col] # Focus on Cohere for now
    # You would add F1 columns of your other models here if they are in df_detailed
    # e.g., f1_columns.extend(['Command-A_F1', 'Finetuned_Model_F1'])
    # Check if these columns exist from the original CSV and merge them if needed for a complete summary.

    # For now, let's extract existing model scores if their columns are consistently named
    # Example: If original CSV has 'Command-A Re-parsed F1', 'Finetuned Model F1'
    potential_original_f1_cols = {
        col_name: col_name.replace(" Re-parsed F1", "_F1").replace(" Model F1", "_F1").replace(" ", "_")
        for col_name in df_full.columns if "F1" in col_name and "Cohere" not in col_name
    }
    original_f1_data_to_merge = df_full[['Prompt'] + list(potential_original_f1_cols.keys())].drop_duplicates(subset=['Prompt'])
    original_f1_data_to_merge.rename(columns=potential_original_f1_cols, inplace=True)
    
    df_detailed_with_all_f1 = pd.merge(df_detailed, original_f1_data_to_merge, on='Prompt', how='left')
    
    all_model_f1_columns = [col for col in df_detailed_with_all_f1.columns if col.endswith('_F1')]
    
    if not all_model_f1_columns:
        print("Warning: No F1 score columns found for aggregation. Cannot generate summary or charts.")
    else:
        model_summary_scores = df_detailed_with_all_f1[all_model_f1_columns].mean().reset_index()
        model_summary_scores.columns = ['Model', 'Average_F1_Score']
        # Clean up model names for display
        model_summary_scores['Model'] = model_summary_scores['Model'].str.replace('_F1', '')
        
        print("\n--- Model Performance Summary (Average F1-Score) ---")
        print(model_summary_scores.sort_values(by='Average_F1_Score', ascending=False))
        print(f"Saving summary scores to: {args.output_summary_csv}")
        model_summary_scores.to_csv(args.output_summary_csv, index=False)

        # --- 5. Visualization ---
        if not os.path.exists(args.output_chart_dir):
            os.makedirs(args.output_chart_dir)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Average_F1_Score', y='Model', data=model_summary_scores.sort_values(by='Average_F1_Score', ascending=False), palette="viridis")
        plt.title('Average F1-Score by Model')
        plt.xlabel('Average F1-Score')
        plt.ylabel('Model')
        plt.tight_layout()
        chart_path = os.path.join(args.output_chart_dir, "average_f1_scores_by_model.png")
        plt.savefig(chart_path)
        print(f"Saved F1 score comparison chart to: {chart_path}")
        plt.close()

    # --- Update JSON Report (Optional - adding scores to existing Cohere entries) ---
    # This part assumes INPUT_JSON_REPORT_PATH contains the list of records from run_cohere_evaluation.py
    print(f"\nAttempting to update JSON report: {args.json_report} with scores...")
    try:
        with open(args.json_report, 'r') as f:
            json_data_list = json.load(f)
        
        if isinstance(json_data_list, list) and not df_cohere_scored.empty:
            # Create a quick lookup for scores
            score_lookup = {}
            for _, row in df_cohere_scored.iterrows():
                # model_name in df_cohere_scored is already prefixed with "Cohere_"
                score_lookup[(row['prompt_text'], row['model_name'])] = {
                    'parsed_deductions': row['parsed_deductions'],
                    'Precision': row['Precision'],
                    'Recall': row['Recall'],
                    'F1-score': row['F1-score'],
                    'ExactMatch': row['ExactMatch']
                }
            
            for record in json_data_list:
                if record.get("model_provider") == "Cohere":
                    # Match using prompt_text and the prefixed model_name
                    key = (record.get("prompt_text"), f"Cohere_{record.get('model_name')}")
                    if key in score_lookup:
                        record.update(score_lookup[key])
            
            with open(args.json_report, 'w') as f:
                json.dump(json_data_list, f, indent=2)
            print(f"JSON report updated with scores at: {args.json_report}")
        else:
            print("JSON report is not a list or no Cohere scores to update it with. Skipping JSON update.")

    except FileNotFoundError:
        print(f"JSON report not found at {args.json_report}. Skipping update.")
    except Exception as e:
        print(f"Error updating JSON report: {e}")

    print("\nProcessing and visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process evaluation results, calculate scores, and generate visualizations.")
    parser.add_argument("--csv_report", default=INPUT_CSV_REPORT_PATH, help="Path to the input CSV report file.")
    parser.add_argument("--json_report", default=INPUT_JSON_REPORT_PATH, help="Path to the input/output JSON report file for score updates.")
    parser.add_argument("--output_detailed_csv", default=OUTPUT_DETAILED_CSV_PATH, help="Path to save the detailed CSV with all scores.")
    parser.add_argument("--output_summary_csv", default=OUTPUT_SUMMARY_CSV_PATH, help="Path to save the summary CSV of model scores.")
    parser.add_argument("--output_chart_dir", default=OUTPUT_CHART_DIR, help="Directory to save generated charts.")
    # Add argument for ground truth column name if needed

    args = parser.parse_args()
    main(args) 