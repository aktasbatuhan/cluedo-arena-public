#!/usr/bin/env python3
import os
import csv
import json
import cohere
import argparse
import pandas as pd # For easier CSV and data manipulation
from datetime import datetime

# --- Configuration ---
COHERE_MODEL_LIST = [
    "command-a-03-2025",
    "command-r7b-12-2024",
    "command-r-plus-04-2024",
    "c4ai-aya-expanse-8b",
    "c4ai-aya-expanse-32b"
]

# Default report file paths (can be overridden by command-line arguments)
DEFAULT_CSV_REPORT_PATH = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/evaluation_report.csv"
DEFAULT_JSON_REPORT_PATH = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/evaluation_report.json"

# Column name in the CSV that contains the prompts
# This might need to be adjusted based on the actual CSV structure
PROMPT_COLUMN_CSV = "Prompt" 

# --- Helper Functions ---

def initialize_cohere_client():
    """Initializes and returns the Cohere client."""
    try:
        # Ensure COHERE_API_KEY environment variable is set
        if not os.getenv("COHERE_API_KEY"):
            print("Error: COHERE_API_KEY environment variable not set.")
            print("Please set it: export COHERE_API_KEY='<YOUR_KEY>'")
            return None
        # The user example used ClientV2, but the library might just be Client.
        # cohere.Client() is standard for recent versions.
        client = cohere.Client()
        print("Cohere client initialized successfully.")
        return client
    except Exception as e:
        print(f"Error initializing Cohere client: {e}")
        return None

def get_cohere_response(client, model_name, prompt_text):
    """Sends a prompt to a specified Cohere model and returns the response content."""
    try:
        print(f"  Querying Cohere model: {model_name}...")
        response = client.chat(
            model=model_name,
            message=prompt_text,
            # Add other parameters like temperature, max_tokens if needed
        )
        
        # For NonStreamedChatResponse, the text is usually in response.text
        if hasattr(response, 'text') and isinstance(response.text, str):
             return response.text
        
        # Fallback for the structure you initially provided, in case it varies by model/response type
        # This was: response.message.content which is a list of content blocks.
        # However, the error shows 'NonStreamedChatResponse' has no 'message' attribute.
        # The user's example JSON showed a structure like: response.message.content[0].text
        # This suggests that if the 'message' attribute *did* exist, it would be an object itself.
        # Given the current error, we will prioritize response.text
        # If we still see issues, we might need to inspect the raw 'response' object for different models.

        print(f"  Warning: Could not extract text from response structure from {model_name}. Response: {response}")
        return None
    except Exception as e:
        print(f"  Error querying Cohere model {model_name}: {e}")
        return None

def load_prompts_from_csv(csv_filepath, prompt_column):
    """Loads prompts from a specified column in a CSV file."""
    prompts = []
    try:
        df = pd.read_csv(csv_filepath)
        if prompt_column not in df.columns:
            print(f"Error: Prompt column '{prompt_column}' not found in {csv_filepath}.")
            print(f"Available columns: {df.columns.tolist()}")
            return []
        # Assuming we want unique prompts to avoid redundant API calls if prompts are repeated
        prompts = df[prompt_column].unique().tolist()
        print(f"Loaded {len(prompts)} unique prompts from {csv_filepath} (column: '{prompt_column}').")
        return prompts
    except FileNotFoundError:
        print(f"Error: CSV report file not found at {csv_filepath}")
        return []
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return []

def update_csv_report(csv_filepath, new_evaluation_data):
    """Appends new evaluation data to the CSV report."""
    try:
        # Read existing CSV or create a new DataFrame if it doesn't exist
        if os.path.exists(csv_filepath):
            df_existing = pd.read_csv(csv_filepath)
        else:
            df_existing = pd.DataFrame()
        
        df_new = pd.DataFrame(new_evaluation_data)
        
        # Concatenate new data
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Save updated DataFrame to CSV
        df_updated.to_csv(csv_filepath, index=False)
        print(f"CSV report updated successfully at {csv_filepath}")
    except Exception as e:
        print(f"Error updating CSV report {csv_filepath}: {e}")

def update_json_report(json_filepath, new_evaluation_data):
    """Updates the JSON report with new evaluation data."""
    try:
        existing_data = []
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        print(f"Warning: Existing JSON report at {json_filepath} is not a list. It will be overwritten as a list.")
                        existing_data = []
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {json_filepath}. It will be overwritten.")
                    existing_data = []
        
        # Append new data (assuming a list of records)
        updated_data = existing_data + new_evaluation_data
        
        with open(json_filepath, 'w') as f:
            json.dump(updated_data, f, indent=2)
        print(f"JSON report updated successfully at {json_filepath}")
    except Exception as e:
        print(f"Error updating JSON report {json_filepath}: {e}")

# --- Main Logic ---
def main(args):
    cohere_client = initialize_cohere_client()
    if not cohere_client:
        return

    # Determine file paths from arguments or defaults
    csv_report_path = args.csv_report if args.csv_report else DEFAULT_CSV_REPORT_PATH
    json_report_path = args.json_report if args.json_report else DEFAULT_JSON_REPORT_PATH
    prompt_column = args.prompt_column if args.prompt_column else PROMPT_COLUMN_CSV

    print(f"Reading prompts from CSV: {csv_report_path}")
    prompts_to_evaluate = load_prompts_from_csv(csv_report_path, prompt_column)

    if not prompts_to_evaluate:
        print("No prompts to evaluate. Exiting.")
        return

    all_new_results = []
    evaluation_timestamp = datetime.now().isoformat()

    print(f"\nStarting Cohere API evaluations for {len(prompts_to_evaluate)} prompts across {len(COHERE_MODEL_LIST)} models...")

    for i, prompt in enumerate(prompts_to_evaluate):
        print(f"\nProcessing prompt {i+1}/{len(prompts_to_evaluate)}: '{prompt[:100]}...'")
        for model_name in COHERE_MODEL_LIST:
            response_text = get_cohere_response(cohere_client, model_name, prompt)
            if response_text is not None:
                result_entry = {
                    "prompt_id": i, # Simple ID for this run, or use an existing ID from CSV if available
                    "prompt_text": prompt,
                    "model_provider": "Cohere",
                    "model_name": model_name,
                    "generated_response": response_text,
                    "evaluation_timestamp": evaluation_timestamp
                    # Add other fields like tokens, ground_truth if available/needed
                }
                all_new_results.append(result_entry)
            else:
                print(f"  No response or error for model {model_name} on prompt: '{prompt[:50]}...'")
    
    if not all_new_results:
        print("\nNo new results obtained from Cohere API. Reports will not be updated.")
        return

    print(f"\nFinished Cohere API evaluations. {len(all_new_results)} new results obtained.")
    
    print("\nUpdating reports...")
    update_csv_report(csv_report_path, all_new_results)
    update_json_report(json_report_path, all_new_results)

    print("\nEvaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluations using Cohere API and update reports.")
    parser.add_argument("--csv_report", type=str, help=f"Path to the input/output CSV report file. Defaults to {DEFAULT_CSV_REPORT_PATH}")
    parser.add_argument("--json_report", type=str, help=f"Path to the input/output JSON report file. Defaults to {DEFAULT_JSON_REPORT_PATH}")
    parser.add_argument("--prompt_column", type=str, help=f"Name of the column in the CSV file that contains the prompts. Defaults to '{PROMPT_COLUMN_CSV}'.")
    # Potentially add arguments for API key if not using env var, or for Cohere model list

    args = parser.parse_args()
    main(args) 