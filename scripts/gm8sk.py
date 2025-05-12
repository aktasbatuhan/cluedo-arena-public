#!/usr/bin/env python3
import os
import json
import csv
import argparse
import time
from datetime import datetime
import re
import pandas as pd
import random

# Provider-specific libraries
from predibase import Predibase
import cohere
from openai import OpenAI

# For plotting (optional, can be run separately or included)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
COHERE_MODEL_LIST = [
    "command-a-03-2025",
    "command-r7b-12-2024",
    "command-r-plus-04-2024",
    "c4ai-aya-expanse-8b",
    "c4ai-aya-expanse-32b"
]

OPENROUTER_MODEL_LIST = [
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-preview",
    "openai/gpt-4o-mini",
    "qwen/qwen-2.5-7b-instruct"
]

DEFAULT_INPUT_CSV_PATH = "data/gsm8k_sample_100.csv" # From previous step
DEFAULT_OUTPUT_DIR = "eval_reports/gsm8k_eval_reports/"

# --- API Client Initialization ---
def initialize_clients(args):
    clients = {}
    # Predibase
    predibase_api_key = os.getenv("PREDIBASE_API_KEY")
    if not predibase_api_key:
        print("Warning: PREDIBASE_API_KEY environment variable not set. Predibase models will be skipped.")
    else:
        try:
            clients["predibase"] = Predibase(api_token=predibase_api_key)
            print("Predibase client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Predibase client: {e}. Predibase models will be skipped.")

    # Cohere
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        print("Warning: COHERE_API_KEY environment variable not set. Cohere models will be skipped.")
    else:
        try:
            clients["cohere"] = cohere.Client(api_key=cohere_api_key)
            print("Cohere client initialized successfully.")
        except Exception as e:
            print(f"Error initializing Cohere client: {e}. Cohere models will be skipped.")

    # OpenRouter
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Warning: OPENROUTER_API_KEY environment variable not set. OpenRouter models will be skipped.")
    else:
        try:
            clients["openrouter"] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_api_key,
            )
            print("OpenRouter client initialized successfully.")
        except Exception as e:
            print(f"Error initializing OpenRouter client: {e}. OpenRouter models will be skipped.")
    return clients

# --- Helper Functions ---

def extract_gsm8k_ground_truth_answer(answer_str: str) -> str | None:
    """Extracts the final numerical answer from the GSM8K answer string."""
    if not isinstance(answer_str, str):
        return None
    match = re.search(r"####\s*([\d,.-]+)", answer_str)
    if match:
        # Normalize: remove commas, strip whitespace
        gt_val = match.group(1).replace(",", "").strip()
        # Convert "X.0" to "X" for integers
        try:
            num_float = float(gt_val)
            if num_float == int(num_float):
                return str(int(num_float))
        except ValueError:
            pass # Not a float, or other issue
        return gt_val
    return None

def extract_predicted_numerical_answer(model_completion: str, debug_mode: bool = False) -> str | None:
    """Extracts the last numerical value from the model's completion string."""
    if not isinstance(model_completion, str):
        return None

    # Regex to find numbers (integers or decimals, possibly with sign and commas)
    # Handles numbers like: 123, 1,234, 123.45, .5, -10, +5.5
    numbers_found = re.findall(r"[-+]?(?:[\d,]+\.?\d*|\.\d+)", model_completion)

    if debug_mode:
        print(f"  [DEBUG NUM_EXTRACT] Raw completion: \"{model_completion[:100]}...\"")
        print(f"  [DEBUG NUM_EXTRACT] Numbers found by regex: {numbers_found}")

    if not numbers_found:
        return None

    # Take the last number found, as models often conclude with the answer
    last_number_str = numbers_found[-1]

    # Normalize: remove commas
    normalized_number_str = last_number_str.replace(",", "").strip()

    # Further normalization: if it's like "123.0" or "123.00", convert to "123"
    # This helps match integers correctly if the model outputs a float ending in .0
    try:
        num_float = float(normalized_number_str)
        if num_float == int(num_float): # Check if it's a whole number
            final_answer = str(int(num_float))
            if debug_mode: print(f"  [DEBUG NUM_EXTRACT] Normalized to int string: {final_answer}")
            return final_answer
    except ValueError:
        # Not a valid float, proceed with the string as is (after comma removal)
        if debug_mode: print(f"  [DEBUG NUM_EXTRACT] Not a float or no int conversion: {normalized_number_str}")
        pass
    
    if debug_mode: print(f"  [DEBUG NUM_EXTRACT] Final extracted number string: {normalized_number_str}")
    return normalized_number_str


def calculate_exact_match(predicted_answer_str: str | None, ground_truth_answer_str: str | None) -> float:
    """Calculates exact match between two normalized numerical strings."""
    if predicted_answer_str is None or ground_truth_answer_str is None:
        return 0.0
    return 1.0 if predicted_answer_str == ground_truth_answer_str else 0.0

def generate_gsm8k_prompt(question: str) -> str:
    """Creates a prompt for GSM8K questions."""
    return f"Question: {question}\\n\\nPlease provide the final numerical answer only.\\nAnswer:"

# --- Main Evaluation Logic ---
def main(args):
    print("Starting GSM8K evaluation script...")
    clients = initialize_clients(args)

    # --- Load Data ---
    try:
        df_gsm8k = pd.read_csv(args.input_csv_path)
        # Ensure we take at most num_eval_samples if specified and available
        if args.num_eval_samples > 0 and args.num_eval_samples < len(df_gsm8k):
            print(f"Sampling {args.num_eval_samples} questions from {args.input_csv_path}.")
            df_gsm8k = df_gsm8k.sample(n=args.num_eval_samples, random_state=args.random_seed)
        elif args.num_eval_samples > 0:
             print(f"Requested {args.num_eval_samples} samples, but only {len(df_gsm8k)} available. Using all available.")
        else:
            print(f"Using all {len(df_gsm8k)} questions from {args.input_csv_path} (sampling disabled or not needed).")

        prompts_data = []
        for idx, row in df_gsm8k.iterrows():
            question = row.get("question")
            answer_raw = row.get("answer")
            if pd.isna(question) or pd.isna(answer_raw):
                print(f"Warning: Skipping row {idx} due to missing question or answer.")
                continue
            
            ground_truth_ans = extract_gsm8k_ground_truth_answer(answer_raw)
            if ground_truth_ans is None:
                print(f"Warning: Could not extract ground truth from row {idx}. Answer: \"{str(answer_raw)[:50]}...\"")
                continue
                
            prompts_data.append({
                "id": f"gsm8k_sample_{idx}",
                "question_text": question,
                "ground_truth_answer_str": ground_truth_ans
            })
        
        if not prompts_data:
            print("No valid prompts loaded. Exiting.")
            return
        print(f"Loaded {len(prompts_data)} questions for evaluation.")

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {args.input_csv_path}")
        return
    except Exception as e:
        print(f"Error reading or processing input CSV file: {e}")
        return

    all_results = []
    evaluation_timestamp = datetime.now().isoformat()
    
    live_scores_aggregator = {} # { "model_full_name": {"em_sum": 0, "count": 0} }


    # --- Main Loop for Evaluation ---
    for prompt_idx, prompt_entry in enumerate(prompts_data):
        prompt_id = prompt_entry["id"]
        question_text = prompt_entry["question_text"]
        ground_truth_answer_str = prompt_entry["ground_truth_answer_str"]
        
        current_prompt = generate_gsm8k_prompt(question_text)
        
        print(f"\\nProcessing prompt {prompt_idx + 1}/{len(prompts_data)}: {prompt_id} ('{question_text[:70]}...')")
        print(f"  Ground Truth Answer: {ground_truth_answer_str}")

        # --- Predibase Evaluation ---
        if args.run_predibase and "predibase" in clients:
            model_full_name_pb = f"Predibase_{args.predibase_adapter_id}"
            if model_full_name_pb not in live_scores_aggregator:
                live_scores_aggregator[model_full_name_pb] = {"em_sum": 0.0, "count": 0}
            
            print(f"  Calling Predibase model (Adapter: {args.predibase_adapter_id})...")
            pb_client = clients["predibase"]
            try:
                base_model_deployment_client = pb_client.deployments.client(args.predibase_deployment_name)
                response = base_model_deployment_client.generate(
                    current_prompt, 
                    adapter_id=args.predibase_adapter_id, 
                    max_new_tokens=args.max_new_tokens_gsm8k # Use specific max tokens for GSM8K
                )
                raw_response = response.generated_text
                predicted_answer_str = extract_predicted_numerical_answer(raw_response, args.debug_num_extract)
                em = calculate_exact_match(predicted_answer_str, ground_truth_answer_str)
                
                live_scores_aggregator[model_full_name_pb]["em_sum"] += em
                live_scores_aggregator[model_full_name_pb]["count"] += 1
                
                all_results.append({
                    "prompt_id": prompt_id, "question_text": question_text, 
                    "ground_truth_answer": ground_truth_answer_str, "model_provider": "Predibase", 
                    "model_name": args.predibase_adapter_id, "raw_response": raw_response,
                    "predicted_answer": predicted_answer_str, "exact_match": em, 
                    "timestamp": evaluation_timestamp, "error_message": ""
                })
                print(f"    Predibase -> Predicted: {predicted_answer_str}, Exact Match: {em:.1f}")
            except Exception as e:
                print(f"    Error with Predibase model {args.predibase_adapter_id}: {e}")
                live_scores_aggregator[model_full_name_pb]["count"] += 1 
                all_results.append({
                    "prompt_id": prompt_id, "question_text": question_text,
                    "ground_truth_answer": ground_truth_answer_str, "model_provider": "Predibase", 
                    "model_name": args.predibase_adapter_id, "raw_response": "ERROR",
                    "predicted_answer": None, "exact_match": 0.0, 
                    "timestamp": evaluation_timestamp, "error_message": str(e)
                })
            time.sleep(args.api_call_delay)

        # --- Cohere Evaluation ---
        if args.run_cohere and "cohere" in clients:
            cohere_client = clients["cohere"]
            for model_name in COHERE_MODEL_LIST:
                model_full_name_co = f"Cohere_{model_name}"
                if model_full_name_co not in live_scores_aggregator:
                    live_scores_aggregator[model_full_name_co] = {"em_sum": 0.0, "count": 0}

                print(f"  Calling Cohere model: {model_name}...")
                try:
                    response = cohere_client.chat(
                        model=model_name,
                        message=current_prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_new_tokens_gsm8k
                    )
                    raw_response = response.text if hasattr(response, 'text') else str(response)
                    predicted_answer_str = extract_predicted_numerical_answer(raw_response, args.debug_num_extract)
                    em = calculate_exact_match(predicted_answer_str, ground_truth_answer_str)

                    live_scores_aggregator[model_full_name_co]["em_sum"] += em
                    live_scores_aggregator[model_full_name_co]["count"] += 1

                    all_results.append({
                        "prompt_id": prompt_id, "question_text": question_text,
                        "ground_truth_answer": ground_truth_answer_str, "model_provider": "Cohere", 
                        "model_name": model_name, "raw_response": raw_response,
                        "predicted_answer": predicted_answer_str, "exact_match": em, 
                        "timestamp": evaluation_timestamp, "error_message": ""
                    })
                    print(f"    Cohere ({model_name}) -> Predicted: {predicted_answer_str}, Exact Match: {em:.1f}")
                except Exception as e:
                    print(f"    Error with Cohere model {model_name}: {e}")
                    live_scores_aggregator[model_full_name_co]["count"] += 1
                    all_results.append({
                        "prompt_id": prompt_id, "question_text": question_text,
                        "ground_truth_answer": ground_truth_answer_str, "model_provider": "Cohere", 
                        "model_name": model_name, "raw_response": "ERROR",
                        "predicted_answer": None, "exact_match": 0.0, 
                        "timestamp": evaluation_timestamp, "error_message": str(e)
                    })
                time.sleep(args.api_call_delay)

        # --- OpenRouter Evaluation ---
        if args.run_openrouter and "openrouter" in clients:
            openrouter_client = clients["openrouter"]
            for model_name in OPENROUTER_MODEL_LIST:
                model_full_name_or = f"OpenRouter_{model_name}"
                if model_full_name_or not in live_scores_aggregator:
                    live_scores_aggregator[model_full_name_or] = {"em_sum": 0.0, "count": 0}
                
                print(f"  Calling OpenRouter model: {model_name}...")
                try:
                    response = openrouter_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=args.temperature,
                        max_tokens=args.max_new_tokens_gsm8k,
                         extra_headers={ # As per OpenRouter docs for free tier
                            "HTTP-Referer": args.openrouter_site_url or "http://localhost",
                            "X-Title": args.openrouter_site_title or "GSM8K Eval"
                        }
                    )
                    raw_response = response.choices[0].message.content
                    predicted_answer_str = extract_predicted_numerical_answer(raw_response, args.debug_num_extract)
                    em = calculate_exact_match(predicted_answer_str, ground_truth_answer_str)

                    live_scores_aggregator[model_full_name_or]["em_sum"] += em
                    live_scores_aggregator[model_full_name_or]["count"] += 1

                    all_results.append({
                        "prompt_id": prompt_id, "question_text": question_text,
                        "ground_truth_answer": ground_truth_answer_str, "model_provider": "OpenRouter", 
                        "model_name": model_name, "raw_response": raw_response,
                        "predicted_answer": predicted_answer_str, "exact_match": em, 
                        "timestamp": evaluation_timestamp, "error_message": ""
                    })
                    print(f"    OpenRouter ({model_name}) -> Predicted: {predicted_answer_str}, Exact Match: {em:.1f}")
                except Exception as e:
                    print(f"    Error with OpenRouter model {model_name}: {e}")
                    live_scores_aggregator[model_full_name_or]["count"] += 1
                    all_results.append({
                        "prompt_id": prompt_id, "question_text": question_text,
                        "ground_truth_answer": ground_truth_answer_str, "model_provider": "OpenRouter", 
                        "model_name": model_name, "raw_response": "ERROR",
                        "predicted_answer": None, "exact_match": 0.0, 
                        "timestamp": evaluation_timestamp, "error_message": str(e)
                    })
                time.sleep(args.api_call_delay)
        
        if (prompt_idx + 1) % args.print_every == 0 and prompt_idx > 0:
            print(f"\\n--- Intermediate Scores after {prompt_idx + 1}/{len(prompts_data)} prompts ---")
            for model_key, scores in live_scores_aggregator.items():
                if scores["count"] > 0:
                    avg_em = scores["em_sum"] / scores["count"]
                    print(f"  Model: {model_key} (evaluated on {scores['count']} prompts)")
                    print(f"    Avg Exact Match: {avg_em:.4f}")
            print("--------------------------------------------------")

    # --- Save Results ---
    if all_results:
        print(f"\\nSaving {len(all_results)} detailed results to CSV: {args.output_csv_path}")
        try:
            keys = all_results[0].keys()
            with open(args.output_csv_path, 'w', newline='', encoding='utf-8') as output_file:
                dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                dict_writer.writeheader()
                dict_writer.writerows(all_results)
            print("Detailed CSV report saved successfully.")
        except Exception as e:
            print(f"Error saving detailed CSV report: {e}")

        print(f"Saving {len(all_results)} detailed results to JSON: {args.output_json_path}")
        try:
            with open(args.output_json_path, 'w', encoding='utf-8') as output_file:
                json.dump(all_results, output_file, indent=2, ensure_ascii=False)
            print("Detailed JSON report saved successfully.")
        except Exception as e:
            print(f"Error saving detailed JSON report: {e}")
        
        # --- Generate Summary Report and Chart ---
        df_results = pd.DataFrame(all_results)
        if not df_results.empty:
            df_results['model_full_name'] = df_results['model_provider'] + "_" + df_results['model_name']
            model_summary_scores = df_results.groupby('model_full_name')['exact_match'].mean().reset_index()
            model_summary_scores.rename(columns={'exact_match': 'Average_ExactMatch', 'model_full_name': 'Model'}, inplace=True)
            
            print("\\n--- Model Performance Summary (Exact Match) ---")
            print(model_summary_scores.sort_values(by='Average_ExactMatch', ascending=False))
            model_summary_scores.to_csv(args.output_summary_csv_path, index=False)
            print(f"Summary scores saved to: {args.output_summary_csv_path}")

            # Generate Chart
            if not os.path.exists(args.output_chart_dir_path):
                os.makedirs(args.output_chart_dir_path)
            
            plt.figure(figsize=(12, max(6, len(model_summary_scores) * 0.5)))
            sns.barplot(x='Average_ExactMatch', y='Model', hue='Model', legend=False, data=model_summary_scores.sort_values(by='Average_ExactMatch', ascending=False), palette="viridis")
            plt.title('Average Exact Match Score by Model (GSM8K)')
            plt.xlabel('Average Exact Match Score')
            plt.ylabel('Model')
            plt.tight_layout()
            chart_path = os.path.join(args.output_chart_dir_path, "average_exact_match_gsm8k.png")
            plt.savefig(chart_path)
            print(f"Saved Exact Match score chart to: {chart_path}")
            plt.close()
    else:
        print("\\nNo results generated to save.")

    print("\\nGSM8K evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSM8K evaluations across multiple LLM providers.")
    
    # File Paths
    parser.add_argument("--input_csv_path", default=DEFAULT_INPUT_CSV_PATH, help="Path to the input CSV file containing GSM8K questions and answers.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save output reports.")
    
    # Provider Toggles
    parser.add_argument("--run_predibase", action="store_true", help="Enable evaluations for Predibase models.")
    parser.add_argument("--run_cohere", action="store_true", help="Enable evaluations for Cohere models.")
    parser.add_argument("--run_openrouter", action="store_true", help="Enable evaluations for OpenRouter models.")
    
    # Predibase Specific
    parser.add_argument("--predibase_adapter_id", default="your_adapter/1", help="Full Predibase adapter ID.")
    parser.add_argument("--predibase_deployment_name", default="qwen2-5-7b-instruct", help="Name of the base model deployment for Predibase.")
    
    # OpenRouter Specific
    parser.add_argument("--openrouter_site_url", default=None, help="Optional HTTP-Referer for OpenRouter (e.g., your app's URL).")
    parser.add_argument("--openrouter_site_title", default=None, help="Optional X-Title for OpenRouter (e.g., your app's name).")
    
    # Common Model Parameters
    parser.add_argument("--max_new_tokens_gsm8k", type=int, default=128, help="Max new tokens for GSM8K model generation (answers are typically short).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 for deterministic for math problems).") 

    # Evaluation Settings
    parser.add_argument("--num_eval_samples", type=int, default=0, help="Number of questions to randomly sample for evaluation. Default: 0 (use all).")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--api_call_delay", type=float, default=1.0, help="Delay in seconds between API calls.")
    parser.add_argument("--debug_num_extract", action="store_true", help="Enable debug prints for numerical answer extraction from LLM responses.")
    parser.add_argument("--print_every", type=int, default=20, help="How often to print intermediate scores.")


    args = parser.parse_args()

    if not (args.run_predibase or args.run_cohere or args.run_openrouter):
        print("Warning: No model providers selected to run (use --run_predibase, --run_cohere, --run_openrouter). Script will only load data if an input path is provided.")
        # If you want to exit if no models are selected:
        # parser.error("No model providers selected. Exiting.")


    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Construct default output file names
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_report_name = f"gsm8k_evaluation_report_{timestamp_str}"
    args.output_csv_path = os.path.join(args.output_dir, f"{base_report_name}_detailed.csv")
    args.output_json_path = os.path.join(args.output_dir, f"{base_report_name}_detailed.json")
    args.output_summary_csv_path = os.path.join(args.output_dir, f"{base_report_name}_summary.csv")
    args.output_chart_dir_path = os.path.join(args.output_dir, "charts") # Subdirectory for charts

    main(args)



