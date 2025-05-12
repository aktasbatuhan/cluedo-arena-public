print("=== DEBUG: Running evaluate_cluedo_model.py version with detailed JSONDecodeError reporting (v_debug_json_err_reporting) ===")
import os
import json
import argparse
import random
from predibase import Predibase
import ast # For literal_eval
import yaml # For parsing model output
import re
import csv # Added import
import pandas as pd # Added import for pandas

# --- Predibase Client Initialization (copied from prompt_predibase_model.py) ---
PREDIBASE_API_KEY = os.getenv("PREDIBASE_API_KEY")
if not PREDIBASE_API_KEY:
    raise ValueError("PREDIBASE_API_KEY environment variable not set.")
try:
    pb = Predibase(api_token=PREDIBASE_API_KEY)
    print("Predibase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Predibase client: {e}")
    exit(1)

# --- Reward Function Logic (adapted from predibase_clue_train.py for evaluation) ---
def extract_yaml_from_completion(completion: str):
    if not completion:  # Handles None or empty string
        return "" 
    text = completion.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[0].startswith('```'): lines = lines[1:]
        if lines and lines[-1].startswith('```'): lines = lines[:-1]
        text = '\n'.join(lines)
    match = re.search(r'(newlyDeducedCards\s*:\s*\n[\s\S]*)', text)
    return match.group(1) if match else text

def get_predicted_deductions(model_completion: str):
    if not model_completion: # Handle None or empty string input
        return set()

    try:
        yaml_text_initial = extract_yaml_from_completion(model_completion)
        if not yaml_text_initial: # If YAML extraction returned nothing (e.g. from None input)
            return set()
        
        # Attempt 1: Load directly if it looks like a complete YAML object
        completion_yaml = yaml.safe_load(yaml_text_initial)
        
        if isinstance(completion_yaml, dict):
            raw_predictions = completion_yaml.get("newlyDeducedCards", [])
            if isinstance(raw_predictions, list):
                # Ensure all items are strings for consistent set comparison
                return set(str(item) for item in raw_predictions if isinstance(item, (str, int, float)))
            else:
                print(f"[Eval Warning] 'newlyDeducedCards' in model output (initial attempt) was not a list: {raw_predictions}. Snippet: {yaml_text_initial[:100]}")
        else:
            print(f"[Eval Warning] Parsed YAML from model (initial attempt) is not a dict. Snippet: {yaml_text_initial[:100]}")

    except yaml.YAMLError as e_initial:
        print(f"[Eval Warning] Initial YAML parse attempt failed: {e_initial}. Model completion snippet: {model_completion[:200]}. Attempting fallback.")
        try:
            # Fallback: Find "newlyDeducedCards:" in the *original* model_completion
            # This helps if extract_yaml_from_completion's regex didn't isolate the block,
            # and the initial parse was on a larger, problematic chunk.
            search_key = "newlyDeducedCards:"
            key_start_index = model_completion.find(search_key)
            
            if key_start_index != -1:
                # Take substring from the found key to the end
                yaml_text_from_key = model_completion[key_start_index:]
                
                # Re-clean this specific segment using extract_yaml_from_completion.
                # This handles potential nested backticks and re-applies the focusing regex.
                cleaned_fallback_text = extract_yaml_from_completion(yaml_text_from_key)
                
                completion_yaml_fb = yaml.safe_load(cleaned_fallback_text)
                
                if isinstance(completion_yaml_fb, dict):
                    raw_predictions_fb = completion_yaml_fb.get("newlyDeducedCards", [])
                    if isinstance(raw_predictions_fb, list):
                        parsed_set_fb = set(str(item) for item in raw_predictions_fb if isinstance(item, (str, int, float)))
                        print(f"[Eval Info] Successfully parsed 'newlyDeducedCards' using fallback. Snippet: {model_completion[:50]}...")
                        return parsed_set_fb
                    else:
                        print(f"[Eval Warning] Fallback 'newlyDeducedCards' in parsed dict was not a list: {raw_predictions_fb}. Fallback text: {cleaned_fallback_text[:100]}")
                else:
                    print(f"[Eval Warning] Fallback parsed YAML (from '{search_key}') is not a dict. Fallback text: {cleaned_fallback_text[:100]}")
            else:
                print(f"[Eval Info] Fallback: Search key '{search_key}' not found in model completion: {model_completion[:200]}")

        except yaml.YAMLError as e_fallback:
            # Log specific error for fallback YAML parse
            fallback_text_snippet_for_log = cleaned_fallback_text[:100] if 'cleaned_fallback_text' in locals() else (yaml_text_from_key[:100] if 'yaml_text_from_key' in locals() else model_completion[:100])
            print(f"[Eval Warning] Fallback YAML parse (from '{search_key}') also failed: {e_fallback}. Fallback text snippet: {fallback_text_snippet_for_log}")
        except Exception as e_fallback_unexpected:
            # Catch any other unexpected errors during fallback
            print(f"[Eval Error] Unexpected error during fallback YAML parsing: {e_fallback_unexpected}. Snippet: {model_completion[:100]}")
            
    except Exception as e_outer:
        # Make snippet printing safe in case model_completion was unexpectedly non-string or became None through complex paths
        completion_snippet = "Error: problem with model_completion variable type" 
        if isinstance(model_completion, str):
            completion_snippet = model_completion[:100]
        elif model_completion is None:
            completion_snippet = "None"
        
        print(f"[Eval Error] Unexpected error parsing model output (outer try): {e_outer}. Completion snippet: {completion_snippet}")
        return set()
    
    return set() # Return empty set if all parsing attempts fail

def parse_ground_truth_string(gt_str):
    """Safely parses ground truth string like '["Card1", "Card2"]' to a set."""
    if not gt_str: return set()
    try:
        evaluated_list = ast.literal_eval(gt_str)
        if isinstance(evaluated_list, list):
            return set(str(item) for item in evaluated_list if isinstance(item, (str, int, float)))
    except (ValueError, SyntaxError, TypeError):
        pass # Error will be handled by returning empty set if it prints warnings
    print(f"[Eval Warning] Could not parse ground_truth_deductions string: '{gt_str}'")
    return set()

def calculate_metrics(predicted_set: set, truth_set: set):
    """Calculates precision, recall, F1, and exact match."""
    if not predicted_set and not truth_set:
        return 1.0, 1.0, 1.0, True # Perfect match for empty sets
    
    tp = len(predicted_set.intersection(truth_set))
    fp = len(predicted_set.difference(truth_set))
    fn = len(truth_set.difference(predicted_set))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = (predicted_set == truth_set)
    return precision, recall, f1, exact_match

# --- Main Evaluation Logic ---
def main(args):
    # Initialize Predibase client (early check for credentials)
    # ... client initialization ...

    print("Starting evaluation...")
    # test_samples = [] # Removed: test_samples will be derived from prompts_data later

    prompts_data = []
    unique_prompts = set()

    if args.input_eval_csv:
        print(f"Processing CSV input file: {args.input_eval_csv}...")
        try:
            df = pd.read_csv(args.input_eval_csv)
            # ... (rest of CSV loading logic, populating prompts_data) ...
            # Ensure it populates prompts_data with dicts containing:
            # "id", "prompt_text", "ground_truth_str"
            # Example:
            # for index, row in df.iterrows():
            #     prompt_text = str(row[args.csv_prompt_column])
            #     if prompt_text not in unique_prompts:
            #         prompts_data.append({
            #             "id": str(row.get(args.csv_id_column, f"csv_row_{index}")),
            #             "prompt_text": prompt_text,
            #             "ground_truth_str": str(row[args.csv_gt_column]),
            #             # "raw_model_completion" was in the previous version, can be kept if useful
            #             # "raw_model_completion": "N/A_FROM_CSV_INPUT"
            #         })
            #         unique_prompts.add(prompt_text)
            # print(f"Loaded {len(prompts_data)} unique prompts from {args.input_eval_csv}.")
            # The apply_model's version of CSV loading already does this correctly.

            # Current CSV loading logic from apply_model output:
            required_cols = [args.csv_prompt_column, args.csv_gt_column]
            if args.csv_id_column not in df.columns:
                print(f"Warning: ID column '{args.csv_id_column}' not found in CSV. Generating sequential IDs.")
                use_sequential_ids = True
            else:
                required_cols.append(args.csv_id_column)
                use_sequential_ids = False

            if not all(col in df.columns for col in required_cols if col != args.csv_id_column or not use_sequential_ids):
                print(f"Error: CSV file must contain columns: {args.csv_prompt_column}, {args.csv_gt_column}" + (f", and {args.csv_id_column}" if not use_sequential_ids else "") + f". Found columns: {df.columns.tolist()}")
                return

            for index, row in df.iterrows():
                prompt_text = str(row[args.csv_prompt_column])
                ground_truth_str_val = str(row[args.csv_gt_column])
                prompt_id_val = str(row[args.csv_id_column]) if not use_sequential_ids else f"csv_row_{index}"

                if prompt_text not in unique_prompts:
                    prompts_data.append({
                        "id": prompt_id_val,
                        "prompt_text": prompt_text,
                        "ground_truth_str": ground_truth_str_val,
                        "raw_model_completion": "N/A_FROM_CSV_INPUT" 
                    })
                    unique_prompts.add(prompt_text)
            print(f"Loaded {len(prompts_data)} unique prompts from {args.input_eval_csv}.")

        except Exception as e:
            print(f"Error processing CSV file {args.input_eval_csv}: {e}")
            return
            
    elif args.input_eval_jsonl:
        print(f"Processing JSONL input file: {args.input_eval_jsonl}...")
        try:
            with open(args.input_eval_jsonl, 'r') as f:
                for i, line in enumerate(f):
                    try: # Inner try for each line
                        data = json.loads(line)
                        prompt_text = data.get("prompt")
                        ground_truth_str_val = data.get("ground_truth_deductions")
                        prompt_id_val = data.get("id", f"jsonl_line_{i}") # Ensured variable name consistency
            
                        if not prompt_text or ground_truth_str_val is None:
                            print(f"Warning: Skipping line {i+1} due to missing 'prompt' or 'ground_truth_deductions': {line.strip()}")
                            continue # Correctly indented within the if, inside inner try

                        # This block is executed if the continue above was not hit
                        if prompt_text not in unique_prompts:
                            prompts_data.append({
                                "id": prompt_id_val,
                                "prompt_text": prompt_text,
                                "ground_truth_str": ground_truth_str_val,
                                "raw_model_completion": "N/A_FROM_JSONL_INPUT"
                            })
                            unique_prompts.add(prompt_text)
                    except json.JSONDecodeError as e_json: # Belongs to inner try
                        print(f"Error decoding JSON from line {i+1}: {e_json}. Line content: {line.strip()}")
                    except Exception as e_line: # Belongs to inner try
                        print(f"Error processing line {i+1} from JSONL: {e_line}. Line content: {line.strip()}")
            print(f"Loaded {len(prompts_data)} unique prompts from {args.input_eval_jsonl}.")
        except FileNotFoundError: # Belongs to outer try for opening file
            print(f"Error: JSONL file not found at {args.input_eval_jsonl}")
            return
        except Exception as e: # Belongs to outer try for opening file
            print(f"Error reading or processing JSONL file {args.input_eval_jsonl}: {e}")
            return

    elif args.llm_interactions_path:
        print(f"Processing llm_interactions file: {args.llm_interactions_path}...")
        try: # Outer try for this elif block
            with open(args.llm_interactions_path, 'r') as f:
                all_entries = json.load(f)
            
            relevant_entries = [
                entry for entry in all_entries 
                if entry.get("type") == "memory_update" and \
                   entry.get("messages") and \
                   isinstance(entry["messages"], list) and \
                   len(entry["messages"]) > 0 and \
                   entry["messages"][0].get("role") == "user" and \
                   entry["messages"][0].get("content") and \
                   entry.get("ground_truth_deductions") is not None # Ensure GT exists
            ]
            print(f"Found {len(relevant_entries)} relevant 'memory_update' entries.")

            for item in relevant_entries:
                raw_user_prompt = item['messages'][0]['content']
                prompt_with_context = CLUE_CONTEXT + raw_user_prompt
                
                if prompt_with_context not in unique_prompts:
                    ground_truth_s = item.get('ground_truth_deductions', '[]') # This is already a string in llm_interactions
                    
                    # Preserve raw_prompt and any existing model outputs if available
                    # These will be used in the conversion to test_samples_all
                    entry_data = {
                        "id": item['id'],
                        "prompt_text": prompt_with_context,
                        "ground_truth_str": ground_truth_s,
                        "raw_prompt_original": raw_user_prompt, # Specific to this source
                        "command_a_raw_output_original": item.get('raw_response_ Command-R+'), # Specific
                        "command_a_parsed_deductions_original": get_predicted_deductions(item.get('raw_response_ Command-R+')) if item.get('raw_response_ Command-R+') else set() # Specific
                    }
                    prompts_data.append(entry_data)
                    unique_prompts.add(prompt_with_context)
            print(f"Loaded {len(prompts_data)} unique prompts from {args.llm_interactions_path}.")

        except FileNotFoundError: # Aligned with the try for llm_interactions_path
            print(f"Error: llm_interactions_path file not found at {args.llm_interactions_path}")
            return # Ensure it returns on error
        except Exception as e: # Aligned with the try for llm_interactions_path
            print(f"Error processing llm_interactions_path file {args.llm_interactions_path}: {e}")
            return # Ensure it returns on error
    else: # Aligned with the main if/elif structure for input file types
        print("No input file specified (use --input_eval_csv, --input_eval_jsonl, or --llm_interactions_path). Exiting.")
        return

    print(f"[DEBUG] Data loading complete. Total unique items in prompts_data: {len(prompts_data)}")

    if not prompts_data:
        print("No data loaded into prompts_data after processing all sources. Exiting.")
        return

    test_samples_all = []
    for p_data in prompts_data:
        try:
            actual_gt_set = parse_ground_truth_string(p_data["ground_truth_str"])
        except Exception as e_gt:
            print(f"Warning: Could not parse ground_truth_str for prompt ID {p_data['id']}: '{p_data['ground_truth_str']}'. Error: {e_gt}. Skipping this prompt.")
            continue

        test_samples_all.append({
            "id": p_data["id"],
            "prompt": p_data["prompt_text"],
            "raw_prompt": p_data.get("raw_prompt_original", p_data["prompt_text"]), # Use original if available, else full prompt
            "actual_ground_truth_set": actual_gt_set,
            "command_a_raw_output": p_data.get("command_a_raw_output_original"), # Use original if available
            "command_a_parsed_deductions_set": p_data.get("command_a_parsed_deductions_original", set()) # Use original if available
        })
    
    print(f"[DEBUG] Converted {len(prompts_data)} items from prompts_data to {len(test_samples_all)} items in test_samples_all (after GT parsing).")

    test_samples = []
    if args.num_eval_samples and len(test_samples_all) > args.num_eval_samples:
        print(f"Sampling {args.num_eval_samples} from {len(test_samples_all)} loaded samples using seed {args.random_seed}...")
        random.seed(args.random_seed)
        try:
            test_samples = random.sample(test_samples_all, args.num_eval_samples)
        except ValueError as e_sample:
            print(f"Error during sampling: {e_sample}. Using all {len(test_samples_all)} samples.")
            test_samples = test_samples_all
    else:
        test_samples = test_samples_all
        if args.num_eval_samples and len(test_samples_all) <= args.num_eval_samples :
             print(f"Number of loaded samples ({len(test_samples_all)}) is less than or equal to num_eval_samples ({args.num_eval_samples}). Using all loaded samples.")
        else: # No num_eval_samples specified or not args.num_eval_samples
            print(f"Using all {len(test_samples_all)} loaded samples (num_eval_samples not specified or <= 0).")


    print(f"[DEBUG] ATTEMPTING TO PROCEED PAST DATA LOADING. Final test_samples length: {len(test_samples)}")
    print(f"[DEBUG] Condition (not test_samples) is {not test_samples}.")

    if not test_samples:
        print("No test samples available for evaluation after processing and sampling. Exiting.")
        return
    
    print(f"\nStarting prompting for {len(test_samples)} test samples...\n")

    # --- Actual Prompting and Evaluation ---
    print("[DEBUG] Before getting Predibase client.") # DEBUG PRINT
    print(f"\nGetting client for deployment: '{args.deployment_name}'") # CHECK 2
    try:
        model_client = pb.deployments.client(args.deployment_name)
        print("Successfully got model client.")
    except Exception as e:
        print(f"Failed to get model client for deployment '{args.deployment_name}': {e}")
        return

    results = []
    all_precisions, all_recalls, all_f1s, all_exact_matches = [], [], [], []
    failed_prompts = 0

    # --- New lists for Command-A re-parsed metrics ---
    all_command_a_reparsed_precisions, all_command_a_reparsed_recalls = [], []
    all_command_a_reparsed_f1s, all_command_a_reparsed_exact_matches = [], []
    # --- End new lists ---

    print(f"\nStarting prompting for {len(test_samples)} test samples...")
    log_interval = max(1, len(test_samples) // 10)

    for i, sample in enumerate(test_samples):
        if (i + 1) % log_interval == 0 or len(test_samples) < log_interval:
            print(f"  Prompting model for sample {i+1}/{len(test_samples)} (ID: {sample['id']})...")
        try:
            response = model_client.generate(
                sample["prompt"],
                adapter_id=args.adapter_id,
                max_new_tokens=args.max_new_tokens
            )
            model_completion = response.generated_text
            predicted_set = get_predicted_deductions(model_completion)
        except Exception as e:
            print(f"    ERROR prompting model for sample ID {sample['id']}: {e}")
            model_completion = "ERROR_DURING_GENERATION"
            predicted_set = set()
            failed_prompts += 1

        # --- Parse Command-A's raw output for deductions ---
        command_a_reparsed_deductions_set = get_predicted_deductions(sample["command_a_raw_output"])
        # --- End Command-A re-parsing ---

        # Metrics for Fine-tuned Model
        ft_precision, ft_recall, ft_f1, ft_exact_match = calculate_metrics(predicted_set, sample["actual_ground_truth_set"]) 
        
        # Metrics for Command-A (Re-parsed)
        ca_re_precision, ca_re_recall, ca_re_f1, ca_re_exact_match = calculate_metrics(command_a_reparsed_deductions_set, sample["actual_ground_truth_set"])

        results.append({
            "id": sample["id"],
            "prompt": sample["prompt"],
            "command_a_raw_output": sample["command_a_raw_output"],
            "command_a_parsed_deductions": list(sample["command_a_parsed_deductions_set"]),
            "command_a_reparsed_deductions": list(command_a_reparsed_deductions_set),
            "finetuned_model_raw_output": model_completion,
            "finetuned_model_deductions": list(predicted_set),
            "ground_truth_deductions": list(sample["actual_ground_truth_set"]),
            
            "command_a_reparsed_precision": ca_re_precision,
            "command_a_reparsed_recall": ca_re_recall,
            "command_a_reparsed_f1": ca_re_f1,
            "command_a_reparsed_exact_match": ca_re_exact_match,

            "finetuned_model_precision": ft_precision,
            "finetuned_model_recall": ft_recall,
            "finetuned_model_f1": ft_f1,
            "finetuned_model_exact_match": ft_exact_match
        })
        # Fine-tuned model metrics aggregation
        all_precisions.append(ft_precision)
        all_recalls.append(ft_recall)
        all_f1s.append(ft_f1)
        if ft_exact_match: all_exact_matches.append(1)
        else: all_exact_matches.append(0)

        # Command-A re-parsed metrics aggregation
        all_command_a_reparsed_precisions.append(ca_re_precision)
        all_command_a_reparsed_recalls.append(ca_re_recall)
        all_command_a_reparsed_f1s.append(ca_re_f1)
        if ca_re_exact_match: all_command_a_reparsed_exact_matches.append(1)
        else: all_command_a_reparsed_exact_matches.append(0)

        if (i + 1) % log_interval == 0 and i + 1 < len(test_samples): 
            print(f"    Metrics calculated for sample {i+1}. Current Finetuned F1 (avg): {sum(all_f1s) / len(all_f1s):.4f}, Current Command-A Reparsed F1 (avg): {sum(all_command_a_reparsed_f1s) / len(all_command_a_reparsed_f1s):.4f}")

    # --- Aggregate and Save Results ---
    print("\n--- Evaluation Summary ---")
    if results:
        avg_precision = sum(all_precisions) / len(all_precisions)
        avg_recall = sum(all_recalls) / len(all_recalls)
        avg_f1 = sum(all_f1s) / len(all_f1s)
        avg_exact_match = sum(all_exact_matches) / len(all_exact_matches)
        
        print(f"Total test samples: {len(test_samples)}")
        print(f"Prompts failed during fine-tuned model generation: {failed_prompts}")
        print("\n--- Fine-tuned Model Metrics ---")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1-Score: {avg_f1:.4f}")
        print(f"Overall Exact Match Accuracy: {avg_exact_match:.4f}")

        # --- Calculate and Print Command-A Re-parsed Metrics ---
        avg_ca_re_precision = sum(all_command_a_reparsed_precisions) / len(all_command_a_reparsed_precisions) if all_command_a_reparsed_precisions else 0.0
        avg_ca_re_recall = sum(all_command_a_reparsed_recalls) / len(all_command_a_reparsed_recalls) if all_command_a_reparsed_recalls else 0.0
        avg_ca_re_f1 = sum(all_command_a_reparsed_f1s) / len(all_command_a_reparsed_f1s) if all_command_a_reparsed_f1s else 0.0
        avg_ca_re_exact_match = sum(all_command_a_reparsed_exact_matches) / len(all_command_a_reparsed_exact_matches) if all_command_a_reparsed_exact_matches else 0.0
        
        print("\n--- Command-A Model (Re-parsed Raw Output) Metrics ---")
        print(f"Average Precision: {avg_ca_re_precision:.4f}")
        print(f"Average Recall: {avg_ca_re_recall:.4f}")
        print(f"Average F1-Score: {avg_ca_re_f1:.4f}")
        print(f"Overall Exact Match Accuracy: {avg_ca_re_exact_match:.4f}")
        # --- End Command-A Re-parsed Metrics ---

        summary_data = {
            "total_samples": len(test_samples),
            "failed_prompts_finetuned_model": failed_prompts,
            "finetuned_model_metrics": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
                "exact_match_accuracy": avg_exact_match
            },
            "command_a_reparsed_metrics": {
                "avg_precision": avg_ca_re_precision,
                "avg_recall": avg_ca_re_recall,
                "avg_f1": avg_ca_re_f1,
                "exact_match_accuracy": avg_ca_re_exact_match
            },
            "detailed_results": results
        }
        try:
            print(f"\nAttempting to save JSON report to: {args.output_report_path}...")
            with open(args.output_report_path, 'w', encoding='utf-8') as f_out:
                json.dump(summary_data, f_out, indent=4, ensure_ascii=False)
            print(f"Detailed report saved to: {args.output_report_path}")

            # --- Write CSV Report ---
            if args.output_csv_path and results:
                print(f"\nAttempting to save CSV report to: {args.output_csv_path}...")
                csv_headers = [
                    "Prompt", 
                    "Command-A Raw Output", 
                    "Finetuned Model Raw Output", 
                    "Command-A Deductions (from parsedOutput)", 
                    "Command-A Deductions (re-parsed from raw)",
                    "Finetuned Model Deductions", 
                    "Ground Truth Deductions",
                    "Command-A Re-parsed Precision",
                    "Command-A Re-parsed Recall",
                    "Command-A Re-parsed F1",
                    "Command-A Re-parsed Exact Match",
                    "Finetuned Model Precision",
                    "Finetuned Model Recall",
                    "Finetuned Model F1",
                    "Finetuned Model Exact Match"
                ]
                try:
                    with open(args.output_csv_path, 'w', encoding='utf-8', newline='') as f_csv:
                        writer = csv.writer(f_csv)
                        writer.writerow(csv_headers)
                        for item in results:
                            writer.writerow([
                                item["prompt"],
                                item["command_a_raw_output"],
                                item["finetuned_model_raw_output"],
                                ", ".join(item["command_a_parsed_deductions"]),
                                ", ".join(item["command_a_reparsed_deductions"]),
                                ", ".join(item["finetuned_model_deductions"]),
                                ", ".join(item["ground_truth_deductions"]),
                                f'{item["command_a_reparsed_precision"]:.4f}',
                                f'{item["command_a_reparsed_recall"]:.4f}',
                                f'{item["command_a_reparsed_f1"]:.4f}',
                                item["command_a_reparsed_exact_match"],
                                f'{item["finetuned_model_precision"]:.4f}',
                                f'{item["finetuned_model_recall"]:.4f}',
                                f'{item["finetuned_model_f1"]:.4f}',
                                item["finetuned_model_exact_match"]
                            ])
                except Exception as e:
                    print(f"Error saving CSV report: {e}")
        except Exception as e:
            print(f"Error saving detailed report: {e}")
    else:
        print("No results to summarize.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Cluedo model on a test set.")
    parser.add_argument("--llm_interactions_path", help="Path to the full llm_interactions.json file. Required if --input_eval_csv is not used.")
    
    # ADDED: Arguments for using a pre-processed CSV as input
    parser.add_argument("--input_eval_csv", type=str, help="Path to an existing evaluation CSV file to source prompts and ground truth from.")
    parser.add_argument("--csv_prompt_column", type=str, default="prompt_text", help="Column name for prompts in the input CSV.")
    parser.add_argument("--csv_gt_column", type=str, default="ground_truth_str", help="Column name for ground truth in the input CSV.")
    parser.add_argument("--csv_id_column", type=str, default="prompt_id", help="Column name for prompt IDs in the input CSV.")
    parser.add_argument("--input_eval_jsonl", type=str, help="Path to an existing evaluation JSONL file to source prompts and ground truth from.") # New argument

    parser.add_argument("--adapter_id", type=str, required=False, help="Predibase adapter ID for fine-tuned model.")
    parser.add_argument("--deployment_name", type=str, default="command-r-plus", help="Predibase deployment name for Command-R+ or other base model.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens for model generation.")
    parser.add_argument("--output_report_path", default="evaluation_report.json", help="Path to save the JSON evaluation report.")
    parser.add_argument("--output_csv_path", default="evaluation_report.csv", help="Path to save the CSV evaluation report.")
    parser.add_argument("--num_eval_samples", type=int, default=0, help="Number of samples to randomly select for evaluation. Set to 0 or negative to use all. Default: 0")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling to ensure reproducibility. Default: 42")
    
    args = parser.parse_args()
    print("\nArguments parsed:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # ADDED: Validate input arguments
    if not args.input_eval_csv and not args.input_eval_jsonl and not args.llm_interactions_path:
        parser.error("Either --llm_interactions_path, --input_eval_csv, or --input_eval_jsonl must be provided.")
    if args.input_eval_csv and args.input_eval_jsonl:
        print("Warning: Both --input_eval_csv and --input_eval_jsonl provided. --input_eval_csv will be used.")

    print("\nCalling main function...")
    main(args) 


    