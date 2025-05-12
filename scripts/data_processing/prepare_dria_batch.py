import json
import uuid
import os
import argparse
import re

CLUE_CONTEXT = (
    "You are an AI agent playing the board game Cluedo (also known as Clue), "
    "a deduction game where players try to determine the suspect, weapon, and room of a crime. "
)

# List of Dria supported models from dria_batch.md
DRIA_MODELS = [
    'claude-3.7-sonnet',
    'claude-3.5-sonnet',
    'gemini-2.5-pro-exp',
    'gemini-2.0-flash',
    'gemma3:4b',
    'gemma3:12b',
    'gemma3:27b',
    'gpt-4o-mini',
    'gpt-4o',
    'llama3.3:70b-instruct-q4_K_M',
    'llama3.1:8b-instruct-q4_K_M',
    'llama3.2:1b-instruct-q4_K_M',
    'mixtral-nemo:12b'
]

def sanitize_model_name_for_filename(model_name):
    """Replaces characters in model name that are problematic for filenames."""
    return re.sub(r'[:/]', '_', model_name)

def main(args):
    try:
        with open(args.input_json_path, 'r', encoding='utf-8-sig') as f:
            all_interactions = json.load(f)
        print(f"Successfully loaded {len(all_interactions)} interactions from {args.input_json_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_json_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {args.input_json_path}: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading {args.input_json_path}: {e}")
        return

    if not isinstance(all_interactions, list):
        print(f"Error: Expected a list of interactions in {args.input_json_path}, but got {type(all_interactions).__name__}.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory set to: {args.output_dir}")

    custom_id_manifest = {}
    # First pass: Collect all valid memory_update prompts and their ground truths
    # This ensures custom_ids are unique across all generated batch files.
    
    valid_prompts_for_batching = []

    print("Processing interactions to extract memory_update prompts and ground truths...")
    skipped_interactions_count = 0

    for i, interaction in enumerate(all_interactions):
        if not isinstance(interaction, dict):
            print(f"  Warning: Skipping non-dict top-level entry at index {i}. Value: {repr(interaction)}")
            skipped_interactions_count += 1
            continue

        if interaction.get("type") == "memory_update":
            interaction_id = interaction.get('timestamp', f"interaction_{i}")

            input_data = interaction.get("input")
            if not isinstance(input_data, dict):
                print(f"  Warning: Skipping 'memory_update' (ID: {interaction_id}) due to invalid 'input' field (expected dict, got {type(input_data).__name__}).")
                skipped_interactions_count += 1
                continue
            
            current_prompt_text = input_data.get("prompt")
            if not isinstance(current_prompt_text, str) or not current_prompt_text.strip():
                print(f"  Warning: Skipping 'memory_update' (ID: {interaction_id}) due to missing or empty 'prompt' in 'input' field.")
                skipped_interactions_count += 1
                continue
            
            contextualized_prompt = CLUE_CONTEXT + current_prompt_text

            parsed_output_data = interaction.get("parsedOutput", {})
            ground_truth_deductions = [] # Default to empty list
            if not isinstance(parsed_output_data, dict):
                 print(f"  Warning: 'parsedOutput' field for 'memory_update' (ID: {interaction_id}) is not a dictionary (got {type(parsed_output_data).__name__}). Ground truth deductions will be empty.")
            else:
                raw_gt_list = parsed_output_data.get("newlyDeducedCards", [])
                if isinstance(raw_gt_list, list):
                    ground_truth_deductions = [str(item) for item in raw_gt_list if isinstance(item, (str, int, float))]
                else:
                    print(f"  Warning: 'newlyDeducedCards' in 'parsedOutput' for 'memory_update' (ID: {interaction_id}) is not a list (got {type(raw_gt_list).__name__}). Ground truth deductions will be empty.")

            custom_id = str(uuid.uuid4())
            custom_id_manifest[custom_id] = {
                "original_prompt": current_prompt_text, # Store original for reference if needed
                "contextualized_prompt": contextualized_prompt,
                "ground_truth_deductions": ground_truth_deductions
            }
            valid_prompts_for_batching.append({
                "custom_id": custom_id,
                "contextualized_prompt": contextualized_prompt
            })
    
    print(f"Found {len(valid_prompts_for_batching)} valid 'memory_update' prompts for Dria batching.")
    if skipped_interactions_count > 0:
        print(f"Skipped {skipped_interactions_count} interactions due to missing/invalid fields or not being 'memory_update' type.")

    if not valid_prompts_for_batching:
        print("No valid prompts found to generate Dria batch files. Exiting.")
        return

    # Generate .jsonl files for each Dria model
    for dria_model_id in DRIA_MODELS:
        sanitized_name = sanitize_model_name_for_filename(dria_model_id)
        output_jsonl_path = os.path.join(args.output_dir, f"dria_batch_{sanitized_name}.jsonl")
        
        with open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
            for prompt_data in valid_prompts_for_batching:
                # Create Dria request payload structure with the new format (messages array)
                dria_request_payload = {
                    "custom_id": prompt_data["custom_id"],
                    "type": "completions",
                    "version": "v1",
                    "body": {
                        "model": dria_model_id,
                        "max_tokens": 1024,
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt_data["contextualized_prompt"]
                            }
                        ]
                    }
                }
                
                # Properly serialize to JSON with ensure_ascii=True to properly escape all control characters
                json_line = json.dumps(dria_request_payload, ensure_ascii=True)
                f_out.write(json_line + '\n')
                
        print(f"  Generated: {output_jsonl_path} with {len(valid_prompts_for_batching)} requests.")

    # Save the manifest file
    manifest_path = os.path.join(args.output_dir, "dria_custom_id_manifest.json")
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f_manifest:
            json.dump(custom_id_manifest, f_manifest, indent=2, ensure_ascii=True)
        print(f"Successfully saved custom ID manifest to: {manifest_path}")
    except Exception as e:
        print(f"Error saving manifest file: {e}")

    print("\nPreparation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare .jsonl batch files for Dria Inference API from Cluedo llm_interactions data.")
    parser.add_argument("--input_json_path", required=True, help="Path to the input JSON file (e.g., data.json) containing all interactions.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the generated .jsonl files and the manifest.json.")
    
    args = parser.parse_args()
    main(args) 