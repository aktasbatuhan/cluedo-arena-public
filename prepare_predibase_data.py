import json
import ast
from pathlib import Path

def format_ground_truth(gt_list):
    """Ensures ground truth is a list of strings, handling None or errors."""
    if gt_list is None:
        return []
    if isinstance(gt_list, str):
        try:
            # Handle cases like "['Card1', 'Card2']"
            # print(f"DEBUG: Attempting literal_eval on string: '{gt_list}'")
            evaluated_list = ast.literal_eval(gt_list)
            if isinstance(evaluated_list, list):
                return [str(item) for item in evaluated_list if isinstance(item, (str, int, float))]
            else:
                print(f"Warning: Ground truth string '{gt_list}' did not evaluate to a list.")
                return []
        except (ValueError, SyntaxError, TypeError) as e:
             # print(f"DEBUG: Error during literal_eval on string: '{gt_list}'. Error: {e}")
             print(f"ERROR during literal_eval on string: '{gt_list}'. Error: {e}")
             return []
    elif isinstance(gt_list, list):
         return [str(item) for item in gt_list if isinstance(item, (str, int, float))]
    else:
        print(f"Warning: Unexpected ground truth type: {type(gt_list)}. Value: {gt_list}")
        return []


input_filename = Path("tiny-grpo/data/cluedo_interactions.jsonl")
output_filename = Path("cluedo_interactions_predibase_grpo.jsonl") 

count = 0
processed_count = 0

try:
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            count += 1
            try:
                data = json.loads(line.strip())

                if data.get("interaction_type") == "memory_update":
                    ground_truth = format_ground_truth(data.get("ground_truth_deductions"))

                    output_record = {
                        "prompt": data.get("prompt", ""),
                        "ground_truth_deductions": json.dumps(ground_truth)
                    }

                    outfile.write(json.dumps(output_record) + '\n')
                    processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")

    print(f"Processing complete.")
    print(f"Total lines read: {count}")
    print(f"Memory update interactions processed: {processed_count}")
    print(f"Output written to: {output_filename.resolve()}")

except FileNotFoundError:
    print(f"Error: Input file not found at {input_filename}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
