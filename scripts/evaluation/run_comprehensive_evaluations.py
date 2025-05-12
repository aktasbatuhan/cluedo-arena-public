#!/usr/bin/env python3
import os
import json
import csv
import argparse
import time
from datetime import datetime
import ast
import yaml
import re
import random # Added for sampling

# ADDED: Imports for data manipulation and plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Provider-specific libraries
from predibase import Predibase
import cohere
from openai import OpenAI

# --- Constants ---
COHERE_MODEL_LIST = [
    "command-a-03-2025", # Placeholder, user confirmed to use list from run_cohere_evaluation.py
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

# MODIFIED: Changed to reflect the new primary input JSON file
DEFAULT_INPUT_JSON_PATH = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/data/llm_interactions.json"
DEFAULT_OUTPUT_DIR = "/Users/batuhanaktas/Desktop/personal/cluedo-arena-public/results/evaluation_reports/"

# ADDED: Context for Cluedo prompts
CLUE_CONTEXT = (
    "You are an AI agent playing the board game Cluedo (also known as Clue), "
    "a deduction game where players try to determine the suspect, weapon, and room of a crime. "
)
    

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

def extract_yaml_from_completion(completion: str, debug_mode: bool = False):
    """
    Extracts a YAML object from a model completion string.
    Handles YAML embedded in markdown code blocks (triple or quadruple backticks) 
    or as a plain string.
    More robust in stripping leading/trailing non-YAML content or markers.
    Now robust to code blocks anywhere in the string.
    """
    if not isinstance(completion, str):
        if debug_mode: print(f"[YAML PARSER DEBUG] Input is not a string: {type(completion)}")
        return None

    completion_cleaned = completion.strip()
    if debug_mode: print(f"[YAML PARSER PRE-REGEX DEBUG] String being fed to regex patterns: <<<\n{completion_cleaned}\n>>>")

    # Flexible regex: find all triple/quadruple backtick code blocks, with or without 'yaml', anywhere in the string
    patterns = [
        re.compile(r"`{4}(?:yaml)?\n([\s\S]*?)\n`{4}", re.DOTALL|re.IGNORECASE), # Quadruple backticks
        re.compile(r"`{3}(?:yaml)?\n([\s\S]*?)\n`{3}", re.DOTALL|re.IGNORECASE)  # Triple backticks
    ]

    extracted_yaml_str = None
    found_blocks = []
    for pattern in patterns:
        found_blocks += pattern.findall(completion_cleaned)
    if debug_mode: print(f"[YAML PARSER DEBUG] Found {len(found_blocks)} code blocks with backticks.")
    # Try to parse each found code block as YAML
    for block in found_blocks:
        block_stripped = block.strip()
        if debug_mode: print(f"[YAML PARSER DEBUG] Trying code block: <<<\n{block_stripped}\n>>>")
        try:
            parsed_yaml = yaml.safe_load(block_stripped)
            if debug_mode: print(f"[YAML PARSER DEBUG] Successfully parsed code block: {parsed_yaml}")
            return parsed_yaml
        except yaml.YAMLError as e:
            if debug_mode: print(f"[YAML PARSER ERROR] Code block safe_load failed: {e}. For content (repr): {repr(block_stripped)}")
            continue

    # Fallback: try to find 'newlyDeducedCards:' section as before
    potential_yaml_start = completion_cleaned.find("newlyDeducedCards:")
    if potential_yaml_start != -1:
        extracted_yaml_str = completion_cleaned[potential_yaml_start:]
        if debug_mode: print(f"[YAML PARSER DEBUG] Extracted by finding 'newlyDeducedCards:': <<<{extracted_yaml_str}>>>")
    else:
        if completion_cleaned.startswith("-") or completion_cleaned.count(":") > 0:
            extracted_yaml_str = completion_cleaned
            if debug_mode: print(f"[YAML PARSER DEBUG] Using cleaned string directly: <<<{extracted_yaml_str}>>>")
        else:
            if debug_mode: print(f"[YAML PARSER DEBUG] No clear YAML content found in: <<<{completion_cleaned}>>>")
            return None

    if not extracted_yaml_str:
        if debug_mode: print(f"[YAML PARSER DEBUG] extracted_yaml_str is empty after attempts.")
        return None

    if debug_mode: print(f"[YAML PARSER PRE-LOAD DEBUG] Attempting to parse: <<<\n{extracted_yaml_str}\n>>>")
    try:
        parsed_yaml = yaml.safe_load(extracted_yaml_str)
        if debug_mode: print(f"[YAML PARSER DEBUG] Successfully parsed: {parsed_yaml}")
        return parsed_yaml
    except yaml.YAMLError as e:
        if debug_mode: print(f"[YAML PARSER ERROR] Initial safe_load failed: {e}. For content (repr): {repr(extracted_yaml_str)}")
        # Fallback 1: Try to find the YAML object starting with newlyDeducedCards more aggressively
        match_inner = re.search(r'(newlyDeducedCards\s*:\s*(?:\n\s*-.*)*)', extracted_yaml_str, re.MULTILINE)
        if match_inner:
            yaml_like_text = match_inner.group(1)
            if debug_mode: print(f"[YAML PARSER DEBUG] Fallback 1: Trying to parse subsection: <<<{yaml_like_text}>>>")
            try:
                parsed_yaml = yaml.safe_load(yaml_like_text)
                if debug_mode: print(f"[YAML PARSER DEBUG] Fallback 1: Successfully parsed subsection: {parsed_yaml}")
                return parsed_yaml # This will be a dict like {"newlyDeducedCards": [...]}
            except yaml.YAMLError as e_inner:
                if debug_mode: print(f"[YAML PARSER ERROR] Fallback 1 (subsection) safe_load also failed: {e_inner}. For content (repr): {repr(yaml_like_text)}")
        # Fallback 2: Improved extraction for list items under newlyDeducedCards
        cards_list = []
        if "newlyDeducedCards:" in extracted_yaml_str:
            try:
                lines = extracted_yaml_str.splitlines()
                found_header = False
                for idx, line in enumerate(lines):
                    if not found_header:
                        if "newlyDeducedCards:" in line:
                            found_header = True
                            # Check if the header and first item are on the same line
                            after_colon = line.split("newlyDeducedCards:", 1)[1]
                            if after_colon.strip().startswith("-"):
                                card_name = after_colon.strip()[1:].strip()
                                if len(card_name) > 1 and ((card_name.startswith('"') and card_name.endswith('"')) or (card_name.startswith("'") and card_name.endswith("'"))):
                                    card_name = card_name[1:-1]
                                if card_name:
                                    cards_list.append(card_name)
                        continue
                    # Only process lines after the header
                    if not line.strip():
                        continue  # skip blank lines
                    if re.match(r"^\s*-", line):
                        card_name = line.strip()[1:].strip()
                        if len(card_name) > 1 and ((card_name.startswith('"') and card_name.endswith('"')) or (card_name.startswith("'") and card_name.endswith("'"))):
                            card_name = card_name[1:-1]
                        if card_name:
                            cards_list.append(card_name)
                    else:
                        # Stop at the first non-list, non-indented, or non-blank line
                        break
                if cards_list:
                    if debug_mode: print(f"[YAML PARSER DEBUG] Fallback 2 (bulletproof): Extracted cards: {cards_list}")
                    return {"newlyDeducedCards": cards_list}
            except Exception as e_regex_fallback:
                if debug_mode: print(f"[YAML PARSER ERROR] Fallback 2 (bulletproof) card extraction failed: {e_regex_fallback}")
        if debug_mode: print(f"[YAML PARSER DEBUG] All parsing attempts failed for: <<<{extracted_yaml_str}>>>\n")
        return None


def get_predicted_deductions(model_completion: str, debug_yaml_parser: bool = False):
    """Extracts and returns a set of deduced cards from the model's completion."""
    parsed_yaml_obj = extract_yaml_from_completion(model_completion, debug_mode=debug_yaml_parser)
    predicted_deductions_set = set()

    if debug_yaml_parser:
        print(f"[DEBUG get_predicted_deductions] parsed_yaml_obj type: {type(parsed_yaml_obj)} value: {parsed_yaml_obj}")

    if isinstance(parsed_yaml_obj, dict):
        raw_predictions = parsed_yaml_obj.get("newlyDeducedCards", [])
        if debug_yaml_parser:
            print(f"[DEBUG get_predicted_deductions] raw_predictions type: {type(raw_predictions)} value: {raw_predictions}")
        if isinstance(raw_predictions, list):
            for item in raw_predictions:
                if isinstance(item, str) and item.strip():
                    predicted_deductions_set.add(item.strip())
        elif isinstance(raw_predictions, str):
            if raw_predictions.strip():
                predicted_deductions_set.add(raw_predictions.strip())
        # else: None or other types are ignored
    # Defensive fallback: if nothing was found, try regex extraction as last resort
    if not predicted_deductions_set:
        # Try to extract lines like '- CardName' after 'newlyDeducedCards:'
        pattern = re.compile(r"newlyDeducedCards:\s*((?:\n\s*-\s*.+)+)", re.MULTILINE)
        match = pattern.search(model_completion)
        if match:
            cards_block = match.group(1)
            for line in cards_block.splitlines():
                line = line.strip()
                if line.startswith("-"):
                    card_name = line[1:].strip()
                    if len(card_name) > 1 and ((card_name.startswith('"') and card_name.endswith('"')) or (card_name.startswith("'") and card_name.endswith("'"))):
                        card_name = card_name[1:-1]
                    if card_name:
                        predicted_deductions_set.add(card_name)
            if debug_yaml_parser:
                print(f"[DEBUG get_predicted_deductions] Fallback regex extracted: {predicted_deductions_set}")

    if debug_yaml_parser:
        raw_completion_snippet = str(model_completion)[:150] if isinstance(model_completion, str) else "Not a string"
        print(f"[DEBUG get_predicted_deductions] Raw model_completion (first 150 chars): {raw_completion_snippet}")
        print(f"[DEBUG get_predicted_deductions] Final predicted_deductions_set to be returned: {predicted_deductions_set}")
    
    return predicted_deductions_set


def parse_ground_truth_jsonl_string(gt_str: str):
    """Safely parses ground truth string like '["Card1", "Card2"]' from JSONL to a set of strings."""
    if not gt_str or not isinstance(gt_str, str):
        return set()
    try:
        # The string from JSONL is already a valid JSON array string
        evaluated_list = json.loads(gt_str) # Use json.loads for JSON array string
        if isinstance(evaluated_list, list):
            # Normalize to set of non-empty, stripped strings
            return set(str(item).strip() for item in evaluated_list if item is not None and str(item).strip())
    except (json.JSONDecodeError, ValueError, SyntaxError, TypeError) as e:
        # print(f"[Eval Warning] Could not parse ground_truth_deductions string: '{gt_str}'. Error: {e}")
        # Fallback for cases where it might be ast.literal_eval compatible but not strict JSON array
        try:
            evaluated_list_ast = ast.literal_eval(gt_str)
            if isinstance(evaluated_list_ast, list):
                return set(str(item).strip() for item in evaluated_list_ast if item is not None and str(item).strip())
        except Exception: # If ast.literal_eval also fails
            # print(f"[Eval Warning] ast.literal_eval also failed for ground_truth_deductions string: '{gt_str}'")
            pass
    return set()


def calculate_metrics(predicted_set: set, truth_set: set):
    """Calculates precision, recall, F1, and exact match."""
    # Ensure inputs are sets of strings
    pred_set = {str(p) for p in predicted_set if isinstance(p, (str, int, float)) and str(p).strip()}
    gt_set = {str(t) for t in truth_set if isinstance(t, (str, int, float)) and str(t).strip()}

    if not gt_set and not pred_set: # Both empty
        return 1.0, 1.0, 1.0, 1.0 # Perfect match for emptiness
    
    # Handle cases where one is empty and the other is not
    # If ground truth is empty, but prediction is not: Precision 0, Recall 1 (no false negatives), F1 0, EM 0
    if not gt_set and pred_set:
        return 0.0, 1.0, 0.0, 0.0
    # If prediction is empty, but ground truth is not: Precision 1 (no false positives), Recall 0, F1 0, EM 0
    if gt_set and not pred_set:
        return 1.0, 0.0, 0.0, 0.0

    true_positives = len(pred_set.intersection(gt_set))
    
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0.0
    # Recall: if gt_set is empty, it's handled by the (not gt_set and not pred_set) or (not gt_set and pred_set) cases.
    # So, if we reach here and gt_set is empty, it implies pred_set is also empty (first case),
    # or pred_set is not empty (second case).
    # If gt_set is non-empty, len(gt_set) > 0.
    recall = true_positives / len(gt_set) if len(gt_set) > 0 else 0.0 
    
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if pred_set == gt_set else 0.0
    
    return precision, recall, f1, exact_match

# --- Main Evaluation Logic ---
def main(args):
    print("Starting comprehensive evaluation script...")
    clients = initialize_clients(args)

    all_extracted_prompts = []
    seen_prompts_for_dedup = set() # Used for deduplication if loading from llm_interactions

    if args.input_jsonl_path:
        print(f"Loading and processing prompts from JSONL file: {args.input_jsonl_path}")
        try:
            with open(args.input_jsonl_path, 'r', encoding='utf-8') as f_jsonl:
                for i, line in enumerate(f_jsonl):
                    try:
                        data = json.loads(line)
                        prompt_text = data.get("prompt")
                        ground_truth_s = data.get("ground_truth_deductions") # This is already a string
                        prompt_id = data.get("id", f"jsonl_line_{i+1}")

                        if not prompt_text or ground_truth_s is None:
                            print(f"Warning: Skipping line {i+1} in JSONL due to missing 'prompt' or 'ground_truth_deductions': {line.strip()}")
                            continue
                        
                        # For JSONL, we assume prompts are unique enough or don't require complex deduplication
                        # If deduplication is needed for JSONL based on prompt_text, it can be added here.
                        # For now, add all valid entries.
                        all_extracted_prompts.append({
                            "id": prompt_id,
                            "text": prompt_text, # Already includes CLUE_CONTEXT if sourced from detailed_report.jsonl
                            "ground_truth_str": ground_truth_s 
                        })
                    except json.JSONDecodeError as e_json:
                        print(f"Error decoding JSON from line {i+1} in {args.input_jsonl_path}: {e_json}. Line: {line.strip()}")
                    except Exception as e_line:
                        print(f"Error processing line {i+1} from {args.input_jsonl_path}: {e_line}. Line: {line.strip()}")
            print(f"Loaded {len(all_extracted_prompts)} prompts from {args.input_jsonl_path}.")
        except FileNotFoundError:
            print(f"Error: Input JSONL file not found at {args.input_jsonl_path}")
            return
        except Exception as e:
            print(f"Error reading or processing input JSONL file: {e}")
            return
    elif args.input_json_path: # Fallback to existing llm_interactions.json logic
        print(f"Loading and processing interactions from: {args.input_json_path}")
        try:
            with open(args.input_json_path, 'r', encoding='utf-8-sig') as f_llm:
                all_interactions = json.load(f_llm)
            print(f"Loaded {len(all_interactions)} total interactions.")

            interaction_id_counter = 0
            memory_update_count = 0
            skipped_due_missing_fields = 0

            for interaction in all_interactions:
                interaction_id_counter += 1
                if not isinstance(interaction, dict):
                    continue

                if interaction.get("type") == "memory_update":
                    memory_update_count += 1
                    input_data = interaction.get("input")
                    if not isinstance(input_data, dict):
                        skipped_due_missing_fields +=1
                        continue
                    
                    current_prompt_text = input_data.get("prompt")
                    if not isinstance(current_prompt_text, str) or not current_prompt_text.strip():
                        skipped_due_missing_fields +=1
                        continue

                    parsed_output_data = interaction.get("parsedOutput", {})
                    raw_gt_list = [] 
                    if isinstance(parsed_output_data, dict):
                        raw_gt_list = parsed_output_data.get("newlyDeducedCards", []) 
                        if raw_gt_list is None: 
                            raw_gt_list = [] 
                    
                    ground_truth_as_string = json.dumps([str(card) for card in raw_gt_list if isinstance(card, (str, int, float))])
                    final_prompt_text = CLUE_CONTEXT + current_prompt_text

                    if final_prompt_text not in seen_prompts_for_dedup:
                        seen_prompts_for_dedup.add(final_prompt_text)
                        all_extracted_prompts.append({
                            "id": interaction.get('timestamp', f'mu_{interaction_id_counter}'),
                            "text": final_prompt_text,
                            "ground_truth_str": ground_truth_as_string
                        })
            
            print(f"Found {memory_update_count} 'memory_update' interactions.")
            print(f"Skipped {skipped_due_missing_fields} 'memory_update' interactions due to missing/invalid prompt/input fields.")
            print(f"Collected {len(all_extracted_prompts)} unique prompts after processing and deduplication from {args.input_json_path}.")

        except FileNotFoundError:
            print(f"Error: Input JSON file not found at {args.input_json_path}")
            return
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {args.input_json_path}: {e.msg} at line {e.lineno} col {e.colno} (pos {e.pos})")
            return
        except Exception as e:
            print(f"Error reading or processing input JSON file: {e}")
            return
    else:
        print("Error: No input file specified. Use --input_jsonl_path or --input_json_path.")
        return

    if not all_extracted_prompts:
        print("No suitable prompts found after processing. Exiting.")
        return

    # --- Sampling Prompts ---
    prompts_data = []
    if args.num_eval_samples > 0 and len(all_extracted_prompts) > args.num_eval_samples:
        print(f"Randomly sampling {args.num_eval_samples} prompts from {len(all_extracted_prompts)} unique prompts.")
        random.seed(args.random_seed)
        prompts_data = random.sample(all_extracted_prompts, args.num_eval_samples)
    else:
        prompts_data = all_extracted_prompts
        if args.num_eval_samples > 0:
            print(f"Requested {args.num_eval_samples} samples, but only {len(all_extracted_prompts)} unique prompts available. Using all available.")
        else:
            print(f"Using all {len(all_extracted_prompts)} unique available prompts (sampling disabled or not needed).")
    
    print(f"Proceeding to evaluate with {len(prompts_data)} prompts.")

    all_results = []
    evaluation_timestamp = datetime.now().isoformat()

    # ADDED: Data structures for accumulating scores for periodic updates
    # { "model_full_name": {"p_sum": 0, "r_sum": 0, "f1_sum": 0, "em_sum": 0, "count": 0} }
    live_scores_aggregator = {}

    # --- Main Loop for Evaluation ---
    for prompt_idx, prompt_entry in enumerate(prompts_data):
        prompt_id = prompt_entry["id"]
        original_prompt_text = prompt_entry["text"] # Store original prompt
        ground_truth_str = prompt_entry["ground_truth_str"]
        
        # --- START PROMPT MODIFICATION ---
        instruction_to_add = (
            "Important: In your response for `newlyDeducedCards`, list only those cards that became definitively known as 'not part of the solution' *during this specific turn's events*. "
            "Do not re-list cards from your hand or other pre-existing knowledge if their status as a deduction was already established *before* this turn's events."
        )

        modified_prompt_text = original_prompt_text # Default to original

        if instruction_to_add not in original_prompt_text: # Avoid duplicating the instruction
            anchor_phrase_before = "Remember: A deduction must be 100% certain."
            anchor_phrase_after = "Based ONLY on the information above, what new cards can you definitively deduce are NOT part of the solution?"
            yaml_instruction_anchor = "Respond ONLY with a YAML object in the following format."
            
            parts_before = original_prompt_text.split(anchor_phrase_before, 1)
            parts_after = original_prompt_text.split(anchor_phrase_after, 1)
            parts_yaml_instruction = original_prompt_text.split(yaml_instruction_anchor, 1)

            inserted = False
            if len(parts_before) == 2: # Preferred: Insert before "Remember:"
                modified_prompt_text = parts_before[0].rstrip() + "\n\n" + instruction_to_add + "\n\n" + anchor_phrase_before + parts_before[1]
                inserted = True
            elif len(parts_after) == 2: # Secondary: Insert after "Based ONLY..."
                # Ensure a newline after the anchor phrase if not present, then add instruction
                modified_prompt_text = parts_after[0] + anchor_phrase_after + "\n" + instruction_to_add + parts_after[1]
                inserted = True
            elif len(parts_yaml_instruction) == 2: # Tertiary: Insert before "Respond ONLY..."
                modified_prompt_text = parts_yaml_instruction[0].rstrip() + "\n\n" + instruction_to_add + "\n\n" + yaml_instruction_anchor + parts_yaml_instruction[1]
                inserted = True
            else: # Fallback: append to the end.
                modified_prompt_text = original_prompt_text + "\n\n" + instruction_to_add
                print(f"[Debug - Prompt {prompt_id}]: Used fallback placement for inserting deduction instruction, appended to end.")
        
        prompt_text = modified_prompt_text # This is the prompt text that will be used by models
        # --- END PROMPT MODIFICATION ---
        
        print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts_data)}: {prompt_id} ('{original_prompt_text[:70]}...')") # Show summary of original
        print(f"Full Prompt Text (with added instruction if applicable):\n{prompt_text}") # Show the potentially modified prompt
        ground_truth_set = parse_ground_truth_jsonl_string(ground_truth_str)

        # --- Predibase Evaluation ---
        if "predibase" in clients:
            model_full_name_pb = f"Predibase_{args.predibase_adapter_id}"
            if model_full_name_pb not in live_scores_aggregator:
                live_scores_aggregator[model_full_name_pb] = {"p_sum": 0.0, "r_sum": 0.0, "f1_sum": 0.0, "em_sum": 0.0, "count": 0}
            
            print(f"  Calling Predibase model (Adapter: {args.predibase_adapter_id} on base deployment: {args.predibase_deployment_name})...")
            pb_client = clients["predibase"]
            try:
                base_model_deployment_client = pb_client.deployments.client(args.predibase_deployment_name)
                response = base_model_deployment_client.generate(
                    prompt_text, 
                    adapter_id=args.predibase_adapter_id, 
                    max_new_tokens=args.max_new_tokens
                )
                raw_response = response.generated_text
                predicted_set = get_predicted_deductions(raw_response, debug_yaml_parser=args.debug_yaml_parser)
                p, r, f1, em = calculate_metrics(predicted_set, ground_truth_set)
                
                live_scores_aggregator[model_full_name_pb]["p_sum"] += p
                live_scores_aggregator[model_full_name_pb]["r_sum"] += r
                live_scores_aggregator[model_full_name_pb]["f1_sum"] += f1
                live_scores_aggregator[model_full_name_pb]["em_sum"] += em
                live_scores_aggregator[model_full_name_pb]["count"] += 1
                
                all_results.append({
                    "prompt_id": prompt_id, "prompt_text": prompt_text, "ground_truth_str": ground_truth_str,
                    "ground_truth_list": sorted(list(ground_truth_set)), "model_provider": "Predibase", 
                    "model_name": args.predibase_adapter_id, "raw_response": raw_response,
                    "parsed_deductions_list": sorted(list(predicted_set)), "precision": p, "recall": r, 
                    "f1_score": f1, "exact_match": em, "timestamp": evaluation_timestamp, "error_message": ""
                })
                print(f"    Predibase F1: {f1:.4f}, Exact Match: {em:.4f}")
                print(f"      Ground Truth: {ground_truth_set}")
                print(f"      Parsed Deductions: {predicted_set}")
                print(f"      Raw Output:\n{raw_response}\n")
            except Exception as e:
                print(f"    Error with Predibase model {args.predibase_adapter_id}: {e}")
                live_scores_aggregator[model_full_name_pb]["count"] += 1 
                all_results.append({
                    "prompt_id": prompt_id, "prompt_text": prompt_text, "ground_truth_str": ground_truth_str,
                    "ground_truth_list": sorted(list(ground_truth_set)), "model_provider": "Predibase", 
                    "model_name": args.predibase_adapter_id, "raw_response": "ERROR",
                    "parsed_deductions_list": [], "precision": 0, "recall": 0, "f1_score": 0, "exact_match": 0, 
                    "timestamp": evaluation_timestamp, "error_message": str(e)
                })
                # Debugging for error case
                print(f"      Ground Truth (at error): {ground_truth_set}")
                print(f"      Parsed Deductions (at error): set()")
                print(f"      Raw Output (at error): ERROR\n")
            time.sleep(args.api_call_delay) 

        # --- Cohere Evaluation ---
        if "cohere" in clients:
            cohere_client = clients["cohere"]
            for model_name in COHERE_MODEL_LIST:
                model_full_name_co = f"Cohere_{model_name}"
                if model_full_name_co not in live_scores_aggregator:
                    live_scores_aggregator[model_full_name_co] = {"p_sum": 0.0, "r_sum": 0.0, "f1_sum": 0.0, "em_sum": 0.0, "count": 0}

                print(f"  Calling Cohere model: {model_name}...")
                try:
                    response = cohere_client.chat(
                        model=model_name,
                        message=prompt_text,
                        temperature=args.temperature,
                    )
                    raw_response = response.text if hasattr(response, 'text') else str(response)
                    predicted_set = get_predicted_deductions(raw_response, debug_yaml_parser=args.debug_yaml_parser)
                    p, r, f1, em = calculate_metrics(predicted_set, ground_truth_set)

                    live_scores_aggregator[model_full_name_co]["p_sum"] += p
                    live_scores_aggregator[model_full_name_co]["r_sum"] += r
                    live_scores_aggregator[model_full_name_co]["f1_sum"] += f1
                    live_scores_aggregator[model_full_name_co]["em_sum"] += em
                    live_scores_aggregator[model_full_name_co]["count"] += 1

                    all_results.append({
                        "prompt_id": prompt_id, "prompt_text": prompt_text, "ground_truth_str": ground_truth_str,
                        "ground_truth_list": sorted(list(ground_truth_set)), "model_provider": "Cohere", 
                        "model_name": model_name, "raw_response": raw_response,
                        "parsed_deductions_list": sorted(list(predicted_set)), "precision": p, "recall": r, 
                        "f1_score": f1, "exact_match": em, "timestamp": evaluation_timestamp, "error_message": ""
                    })
                    print(f"    Cohere ({model_name}) F1: {f1:.4f}, Exact Match: {em:.4f}")
                    print(f"      Ground Truth: {ground_truth_set}")
                    print(f"      Parsed Deductions: {predicted_set}")
                    print(f"      Raw Output:\n{raw_response}\n")
                except Exception as e:
                    print(f"    Error with Cohere model {model_name}: {e}")
                    live_scores_aggregator[model_full_name_co]["count"] += 1
                    all_results.append({
                        "prompt_id": prompt_id, "prompt_text": prompt_text, "ground_truth_str": ground_truth_str,
                        "ground_truth_list": sorted(list(ground_truth_set)), "model_provider": "Cohere", 
                        "model_name": model_name, "raw_response": "ERROR",
                        "parsed_deductions_list": [], "precision": 0, "recall": 0, "f1_score": 0, "exact_match": 0, 
                        "timestamp": evaluation_timestamp, "error_message": str(e)
                    })
                    # Debugging for error case
                    print(f"      Ground Truth (at error): {ground_truth_set}")
                    print(f"      Parsed Deductions (at error): set()")
                    print(f"      Raw Output (at error): ERROR\n")
                time.sleep(1) 

        # --- OpenRouter Evaluation ---
        if "openrouter" in clients:
            openrouter_client = clients["openrouter"]
            for model_name in OPENROUTER_MODEL_LIST:
                model_full_name_or = f"OpenRouter_{model_name}"
                if model_full_name_or not in live_scores_aggregator:
                    live_scores_aggregator[model_full_name_or] = {"p_sum": 0.0, "r_sum": 0.0, "f1_sum": 0.0, "em_sum": 0.0, "count": 0}
                
                print(f"  Calling OpenRouter model: {model_name}...")
                try:
                    response = openrouter_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt_text}],
                        temperature=args.temperature,
                        max_tokens=args.max_new_tokens,
                        extra_headers={
                            "HTTP-Referer": args.openrouter_site_url,
                            "X-Title": args.openrouter_site_title
                        }
                    )
                    raw_response = response.choices[0].message.content
                    predicted_set = get_predicted_deductions(raw_response, debug_yaml_parser=args.debug_yaml_parser)
                    p, r, f1, em = calculate_metrics(predicted_set, ground_truth_set)

                    live_scores_aggregator[model_full_name_or]["p_sum"] += p
                    live_scores_aggregator[model_full_name_or]["r_sum"] += r
                    live_scores_aggregator[model_full_name_or]["f1_sum"] += f1
                    live_scores_aggregator[model_full_name_or]["em_sum"] += em
                    live_scores_aggregator[model_full_name_or]["count"] += 1

                    all_results.append({
                        "prompt_id": prompt_id, "prompt_text": prompt_text, "ground_truth_str": ground_truth_str,
                        "ground_truth_list": sorted(list(ground_truth_set)), "model_provider": "OpenRouter", 
                        "model_name": model_name, "raw_response": raw_response,
                        "parsed_deductions_list": sorted(list(predicted_set)), "precision": p, "recall": r, 
                        "f1_score": f1, "exact_match": em, "timestamp": evaluation_timestamp, "error_message": ""
                    })
                    print(f"    OpenRouter ({model_name}) F1: {f1:.4f}, Exact Match: {em:.4f}")
                    print(f"      Ground Truth: {ground_truth_set}")
                    print(f"      Parsed Deductions: {predicted_set}")
                    print(f"      Raw Output:\n{raw_response}\n")
                except Exception as e:
                    print(f"    Error with OpenRouter model {model_name}: {e}")
                    live_scores_aggregator[model_full_name_or]["count"] += 1
                    all_results.append({
                        "prompt_id": prompt_id, "prompt_text": prompt_text, "ground_truth_str": ground_truth_str,
                        "ground_truth_list": sorted(list(ground_truth_set)), "model_provider": "OpenRouter", 
                        "model_name": model_name, "raw_response": "ERROR",
                        "parsed_deductions_list": [], "precision": 0, "recall": 0, "f1_score": 0, "exact_match": 0, 
                        "timestamp": evaluation_timestamp, "error_message": str(e)
                    })
                    # Debugging for error case
                    print(f"      Ground Truth (at error): {ground_truth_set}")
                    print(f"      Parsed Deductions (at error): set()")
                    print(f"      Raw Output (at error): ERROR\n")
                time.sleep(1) 
        
        # ADDED: Periodic score printing logic
        if (prompt_idx + 1) % 50 == 0 and prompt_idx > 0: # Avoid printing at 0th, print at 50, 100, etc.
            print(f"\n--- Intermediate Scores after {prompt_idx + 1}/{len(prompts_data)} prompts ---")
            for model_key, scores in live_scores_aggregator.items():
                if scores["count"] > 0:
                    avg_p = scores["p_sum"] / scores["count"]
                    avg_r = scores["r_sum"] / scores["count"]
                    avg_f1 = scores["f1_sum"] / scores["count"]
                    avg_em = scores["em_sum"] / scores["count"]
                    print(f"  Model: {model_key} (evaluated on {scores['count']} prompts so far)")
                    print(f"    Avg Precision: {avg_p:.4f}, Avg Recall: {avg_r:.4f}, Avg F1: {avg_f1:.4f}, Avg Exact Match: {avg_em:.4f}")
            print("--------------------------------------------------")

    # --- Save Results ---
    if all_results:
        print(f"\nSaving {len(all_results)} results to CSV: {args.output_csv_path}")
        try:
            # Ensure all dictionaries have the same keys for CSV writing, handle missing keys gracefully if necessary
            # For simplicity, assuming all_results will have consistent keys from the successful appends.
            if not all_results: # Should not happen if we entered this block, but defensive check
                print("No results to write to CSV.")
            else:
                keys = all_results[0].keys() # Get keys from the first result
                with open(args.output_csv_path, 'w', newline='', encoding='utf-8') as output_file:
                    dict_writer = csv.DictWriter(output_file, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(all_results)
                print("CSV report saved successfully.")
        except Exception as e:
            print(f"Error saving CSV report: {e}")

        print(f"\nSaving {len(all_results)} results to JSON: {args.output_json_path}")
        try:
            with open(args.output_json_path, 'w', encoding='utf-8') as output_file:
                json.dump(all_results, output_file, indent=2, ensure_ascii=False)
            print("JSON report saved successfully.")
        except Exception as e:
            print(f"Error saving JSON report: {e}")
    else:
        print("\nNo results generated to save.")

    # ADDED: Call to generate summary and chart if results exist
    if all_results:
        generate_summary_and_chart(all_results, args)

    print("\nComprehensive evaluation script finished.")

# ADDED: Function to generate summary reports and charts
def generate_summary_and_chart(all_results_list, args):
    print("\nGenerating summary reports and charts...")
    df_results = pd.DataFrame(all_results_list)

    # Ensure numeric columns are numeric
    numeric_cols = ['precision', 'recall', 'f1_score', 'exact_match']
    for col in numeric_cols:
        df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

    # --- 1. Create and Save Pivoted Detailed Report ---
    # Combine provider and model name for unique column headers in pivot
    df_results['model_full_name'] = df_results['model_provider'] + "_" + df_results['model_name']
    
    try:
        # metrics_to_pivot = ['f1_score', 'precision', 'recall', 'exact_match', 'parsed_deductions_list']
        # Pivoting with list-like objects (parsed_deductions_list) can be tricky if they are not identical
        # For CSV, it's better to convert lists to strings first if they are to be pivoted directly.
        # However, process_and_visualize_evaluations.py pivoted on numeric scores and then merged parsed_deductions.
        # Let's pivot numeric scores first.
        
        df_pivot_numeric = df_results.pivot_table(
            index=["prompt_id", "prompt_text", "ground_truth_str"], # CHANGED: ground_truth_list to ground_truth_str
            columns='model_full_name',
            values=numeric_cols
        )
        # Flatten multi-index columns (e.g., ('f1_score', 'Cohere_command-r-plus'))
        df_pivot_numeric.columns = [f'{col[1]}_{col[0].replace("-score","").replace("exact_match", "ExactMatch").replace("f1_score","F1")}' for col in df_pivot_numeric.columns.values]
        df_pivot_numeric.reset_index(inplace=True)

        # Now, get the parsed deductions separately and merge if needed, or save as is.
        # For simplicity, the current flat CSV (args.output_csv_path) already has all data including parsed_deductions_list.
        # The pivoted numeric view is good for quick comparison of scores.
        print(f"Saving pivoted detailed numeric scores to: {args.output_detailed_pivot_csv_path}")
        df_pivot_numeric.to_csv(args.output_detailed_pivot_csv_path, index=False)

    except Exception as e:
        print(f"Error creating or saving pivoted detailed CSV: {e}")

    # --- 2. Calculate and Save Summary Scores ---
    try:
        # Group by the combined model_full_name for summary
        model_summary_scores = df_results.groupby('model_full_name')[numeric_cols].mean().reset_index()
        model_summary_scores.rename(columns={
            'model_full_name': 'Model',
            'f1_score': 'Average_F1_Score',
            'precision': 'Average_Precision',
            'recall': 'Average_Recall',
            'exact_match': 'Average_ExactMatch'
        }, inplace=True)

        print("\n--- Model Performance Summary ---")
        print(model_summary_scores.sort_values(by='Average_F1_Score', ascending=False))
        print(f"Saving summary scores to: {args.output_summary_csv_path}")
        model_summary_scores.to_csv(args.output_summary_csv_path, index=False)
    except Exception as e:
        print(f"Error calculating or saving summary scores: {e}")
        return # Cant plot if summary fails

    # --- 3. Generate and Save F1 Score Bar Chart ---
    if not model_summary_scores.empty:
        try:
            if not os.path.exists(args.output_chart_dir_path):
                os.makedirs(args.output_chart_dir_path)
                print(f"Created chart directory: {args.output_chart_dir_path}")
            
            plt.figure(figsize=(12, max(6, len(model_summary_scores) * 0.5))) # Adjust height based on num models
            # MODIFIED: Added hue and legend=False to address FutureWarning
            sns.barplot(x='Average_F1_Score', y='Model', hue='Model', legend=False, data=model_summary_scores.sort_values(by='Average_F1_Score', ascending=False), palette="viridis")
            plt.title('Average F1-Score by Model')
            plt.xlabel('Average F1-Score')
            plt.ylabel('Model')
            plt.tight_layout()
            chart_path = os.path.join(args.output_chart_dir_path, "average_f1_scores_by_model.png")
            plt.savefig(chart_path)
            print(f"Saved F1 score comparison chart to: {chart_path}")
            plt.close()
        except Exception as e:
            print(f"Error generating or saving F1 score chart: {e}")
    else:
        print("No summary scores to plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive evaluations across multiple LLM providers.")
    
    # File Paths
    # MODIFIED: Changed argument name and help text for the input file
    parser.add_argument("--input_json_path", default=None, help="Path to the input JSON file containing interactions (e.g., llm_interactions.json).") # Default to None
    parser.add_argument("--input_jsonl_path", default=None, help="Path to an input JSONL file containing prompts (e.g., prompts_from_detailed_report.jsonl).") # New argument
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save output reports.")
    
    # Provider Toggles - REMOVED
    # parser.add_argument("--run_predibase", action="store_true", help="Enable evaluations for Predibase models.")
    # parser.add_argument("--run_cohere", action="store_true", help="Enable evaluations for Cohere models.")
    # parser.add_argument("--run_openrouter", action="store_true", help="Enable evaluations for OpenRouter models.")
    
    # Predibase Specific
    parser.add_argument("--predibase_adapter_id", default="clue_final_shot/1", help="Full Predibase adapter ID. Defaults to cluedo_memory_grpo_adapter/1.")
    parser.add_argument("--predibase_deployment_name", default="qwen2-5-7b-instruct", help="Name of the base model deployment for Predibase. Defaults to qwen2-5-7b-instruct.")
    
    # OpenRouter Specific
    parser.add_argument("--openrouter_site_url", default="http://localhost", help="Optional HTTP-Referer for OpenRouter.")
    parser.add_argument("--openrouter_site_title", default="Cluedo Arena Eval", help="Optional X-Title for OpenRouter.")
    
    # Common Model Parameters
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens for model generation.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for model generation.") 

    # ADDED: Arguments for sampling
    parser.add_argument("--num_eval_samples", type=int, default=300, help="Number of unique prompts to randomly sample for evaluation. Default: 300")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling to ensure reproducibility. Default: 42")
    parser.add_argument("--api_call_delay", type=float, default=1.0, help="Delay in seconds between API calls to each provider model. Default: 1.0")

    # ADDED: Arguments for summary report and chart output directory
    parser.add_argument("--output_summary_csv_suffix", default="_summary_scores", help="Suffix for the summary scores CSV file.")
    parser.add_argument("--output_charts_subdir", default="charts", help="Subdirectory within output_dir to save charts.")
    parser.add_argument("--output_detailed_pivot_csv_suffix", default="_detailed_pivot", help="Suffix for the pivoted detailed CSV report.")
    parser.add_argument("--debug_yaml_parser", action="store_true", help="Enable detailed debug logging for the YAML parsing process.")

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Construct default output file names
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_report_name = f"comprehensive_evaluation_report_{timestamp_str}"
    args.output_csv_path = os.path.join(args.output_dir, f"{base_report_name}.csv") # Flat detailed report
    args.output_json_path = os.path.join(args.output_dir, f"{base_report_name}.json")
    
    # ADDED: Construct paths for summary and pivoted detailed CSV, and chart directory
    args.output_detailed_pivot_csv_path = os.path.join(args.output_dir, f"{base_report_name}{args.output_detailed_pivot_csv_suffix}.csv")
    args.output_summary_csv_path = os.path.join(args.output_dir, f"{base_report_name}{args.output_summary_csv_suffix}.csv")
    args.output_chart_dir_path = os.path.join(args.output_dir, args.output_charts_subdir)

    if not args.input_json_path and not args.input_jsonl_path:
        parser.error("Either --input_json_path or --input_jsonl_path must be provided.")
    if args.input_json_path and args.input_jsonl_path:
        print("Warning: Both --input_json_path and --input_jsonl_path provided. --input_jsonl_path will be used.")
        args.input_json_path = None # Prioritize JSONL if both given

    main(args) 