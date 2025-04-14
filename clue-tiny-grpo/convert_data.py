import json
import os
from pathlib import Path
import sys

def convert_interactions(input_file: Path, output_file: Path):
    """
    Reads llm_interactions.json, processes interactions, pairs memory updates
    with deduction comparisons, and writes to a JSON Lines file.
    """
    print(f"Starting conversion: {input_file} -> {output_file}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            interactions = json.load(f)
        print(f"Successfully loaded {len(interactions)} interactions from {input_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file}: {e}", file=sys.stderr)
        sys.exit(1)
    except MemoryError:
        print(f"Error: Input file {input_file} is too large to load into memory directly.", file=sys.stderr)
        print("Consider using a streaming JSON parser if memory is insufficient.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_file}: {e}", file=sys.stderr)
        sys.exit(1)

    training_data = []
    # Use index to reliably link memory_update to its deduction_comparison
    last_memory_update_index = {} # agent_name -> index in training_data list

    for i, interaction in enumerate(interactions):
        interaction_type = interaction.get("type")
        agent_name = interaction.get("agent")

        # Default record structure
        record = {
            "interaction_type": interaction_type,
            "prompt": None,
            "chosen_response": None,
            "ground_truth_deductions": None,
            "logged_reward": None
        }

        # Extract common fields
        if interaction.get("input") and isinstance(interaction["input"], dict):
            record["prompt"] = interaction["input"].get("prompt")
        record["chosen_response"] = interaction.get("parsedOutput") or interaction.get("output") # Fallback to raw output

        if interaction_type in ["suggestion", "evaluate_challenge", "consider_accusation"]:
             if record["prompt"] and record["chosen_response"]:
                 training_data.append(record)
             else:
                 print(f"Warning: Skipping interaction {i} (type: {interaction_type}) due to missing prompt or response.")

        elif interaction_type == "memory_update":
            if record["prompt"] and record["chosen_response"]:
                # Add the record first, then store its index
                training_data.append(record)
                if agent_name:
                    last_memory_update_index[agent_name] = len(training_data) - 1 # Store index of this record
            else:
                print(f"Warning: Skipping interaction {i} (type: {interaction_type}) due to missing prompt or response.")

        elif interaction_type == "deduction_comparison":
            if agent_name in last_memory_update_index:
                target_index = last_memory_update_index[agent_name]
                # Check if the record at target_index is indeed the corresponding memory_update
                if target_index < len(training_data) and training_data[target_index]["interaction_type"] == "memory_update":
                    # Update the existing memory_update record
                    training_data[target_index]["ground_truth_deductions"] = interaction.get("groundTruthDeductions")
                    training_data[target_index]["logged_reward"] = interaction.get("reward")
                    # Clear the index to prevent accidental reuse if format is unexpected
                    del last_memory_update_index[agent_name]
                else:
                     print(f"Warning: Mismatched deduction_comparison for agent {agent_name} at index {i}. Expected memory_update at index {target_index}.")
            else:
                print(f"Warning: Found deduction_comparison for agent {agent_name} at index {i} without a preceding memory_update.")
        else:
            print(f"Warning: Skipping interaction {i} with unknown type: {interaction_type}")


    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write the processed data to JSON Lines file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in training_data:
                f.write(json.dumps(record) + '\n')
        print(f"Successfully wrote {len(training_data)} training examples to {output_file}")
    except Exception as e:
        print(f"An error occurred while writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Assuming the script is run from the workspace root or tiny-grpo directory
    base_path = Path(__file__).parent.parent # Assumes script is in tiny-grpo/
    input_json_file = base_path / "llm_interactions.json"
    output_jsonl_file = base_path / "tiny-grpo" / "data" / "cluedo_interactions.jsonl"

    if not input_json_file.is_file():
         # Try looking in workspace root if not found relative to script parent
         input_json_file = Path("llm_interactions.json")
         if not input_json_file.is_file():
              print(f"Error: Cannot find input file llm_interactions.json in {base_path} or workspace root.", file=sys.stderr)
              sys.exit(1)

    convert_interactions(input_json_file, output_jsonl_file)
    print("Conversion complete.") 