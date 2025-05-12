import os
import ast
import yaml  # Requires PyYAML (pip install PyYAML)
from pathlib import Path
from predibase import Predibase, GRPOConfig, RewardFunctionsConfig

# --- Configuration ---
PREDIBASE_API_KEY = os.getenv("PREDIBASE_API_KEY")
if not PREDIBASE_API_KEY:
    raise ValueError("PREDIBASE_API_KEY environment variable not set.")
# !! Set this to the exact name given in the Predibase UI !!
PREDIBASE_DATASET_NAME = "clue_memory_train_data" # Use the confirmed name
# !! Replace with the desired Predibase repository name to save the adapter !!
# Format: "your_org/your_repo_name" or just "your_repo_name"
PREDIBASE_REPO_NAME = "clue_final_shot"
# Base model to fine-tune (ensure it's supported by Predibase RFT)
BASE_MODEL = "qwen2-5-7b-instruct" # Changed based on user request and previous error
# Training Hyperparameters (aligned with previous adjustments)
LEARNING_RATE = 2e-5
ROLLOUTS_PER_PROMPT = 16 # Equivalent to train_group_size
TRAIN_STEPS = 400 # Default is 1000, reduced for potentially faster initial results/cost savings

# --- Reward Function Definition ---
# Note: Imports must be *inside* the function for Predibase execution
def calculate_memory_update_reward(prompt: str, completion: str, example: dict[str, str]) -> float:
    """
    Calculates reward based on correctness of deduced cards in memory updates (expects YAML).
    Now more forgiving: strips code block markers, ignores extra text, and logs parse failures.
    """
    import yaml
    import ast # For converting string representation of list
    import re

    def extract_yaml(text):
        # Remove code block markers if present
        text = text.strip()
        if text.startswith('```'):
            # Remove the first code block line
            lines = text.splitlines()
            if lines[0].startswith('```'):
                lines = lines[1:]
            # Remove last line if it's a code block end
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            text = '\n'.join(lines)
        # Try to extract YAML object from text (find first 'newlyDeducedCards:' and take from there)
        match = re.search(r'(newlyDeducedCards\s*:\s*\n[\s\S]*)', text)
        if match:
            return match.group(1)
        return text

    try:
        # Extract YAML from completion
        yaml_text = extract_yaml(completion)
        completion_yaml = yaml.safe_load(yaml_text)
        if not isinstance(completion_yaml, dict):
            print(f"[Reward Fn Warning] Parsed YAML is not a dict. Completion: {completion[:100]}")
            return 0.0 # Penalize invalid YAML structure immediately

        # --- Safely get raw predictions ---
        raw_predictions = completion_yaml.get("newlyDeducedCards", [])
        predicted_deductions = set()
        if isinstance(raw_predictions, list):
            for item in raw_predictions:
                if isinstance(item, str):
                    predicted_deductions.add(item)
        else:
            print(f"[Reward Fn Warning] 'newlyDeducedCards' was not a list: {raw_predictions}")

        # --- Safely get ground truth from the 'example' dict ---
        ground_truth_str = example.get("ground_truth_deductions")
        truth_set = set()
        if ground_truth_str:
            try:
                ground_truth_list = ast.literal_eval(ground_truth_str)
                if isinstance(ground_truth_list, list):
                    truth_set = set(item for item in ground_truth_list if isinstance(item, str))
                else:
                    print(f"[Reward Fn Warning] 'ground_truth_deductions' field ('{ground_truth_str}') did not evaluate to a list.")
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"[Reward Fn Warning] Could not parse 'ground_truth_deductions' field ('{ground_truth_str}'): {e}")
                truth_set = set()

        if not predicted_deductions and not truth_set:
            return 1.0 # Correctly deduced nothing new when nothing was expected

        intersection = len(predicted_deductions.intersection(truth_set))
        precision = intersection / len(predicted_deductions) if len(predicted_deductions) > 0 else 0.0
        recall = intersection / len(truth_set) if len(truth_set) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Give full reward only if perfect match
        if len(predicted_deductions) == len(truth_set) and intersection == len(truth_set):
            return 1.0
        elif (precision + recall) > 0:
            return f1
        elif not predicted_deductions and truth_set:
            return 0.0 # Predicted nothing, but should have
        elif predicted_deductions and not truth_set:
            return 0.0 # Penalize hallucination
        else:
            return 0.0

    except yaml.YAMLError as e:
        print(f"[Reward Fn Warning] YAML parse error: {e}. Completion: {completion[:100]}")
        return 0.0 # Strict penalty for invalid YAML
    except Exception as e:
        print(f"[Reward Fn Error] Unexpected error calculating reward: {e}. Completion: {completion[:100]}")
        return 0.0


# --- Initialize Predibase Client ---
try:
    # Explicitly pass the api_token from the variable
    pb = Predibase(api_token=PREDIBASE_API_KEY)
    print("Predibase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Predibase client: {e}")
    print("Ensure PREDIBASE_API_KEY environment variable is set correctly.")
    exit(1)

# --- Get the dataset from Predibase by Name ---
print(f"Attempting to retrieve dataset '{PREDIBASE_DATASET_NAME}' from Predibase...")
try:
    # Get the dataset by its name
    dataset = pb.datasets.get(PREDIBASE_DATASET_NAME)
    print(f"Successfully retrieved dataset '{dataset.name}'")
except Exception as e:
    print(f"Error retrieving dataset '{PREDIBASE_DATASET_NAME}': {e}")
    print("Please ensure the dataset name is correct and it exists in your Predibase account.")
    exit(1)

# --- Configure Reward Functions ---
# We only need the memory update reward since we assume the dataset is filtered
reward_config = RewardFunctionsConfig(
    functions={
        "memory_update_f1": calculate_memory_update_reward,
        # Add other reward functions here if needed (e.g., basic YAML format check)
    },
    # If your reward function needs external packages, specify them here:
    # runtime=RewardFunctionsRuntimeConfig(packages=["mypackage"])
)

# --- Configure GRPO Job ---
grpo_config = GRPOConfig(
    base_model=BASE_MODEL,
    reward_fns=reward_config,
    learning_rate=LEARNING_RATE,
    rollouts_per_prompt=ROLLOUTS_PER_PROMPT,
    train_steps=TRAIN_STEPS,
    # prompt_template: By default, Predibase uses the 'prompt' column directly.
    # Ensure your dataset's 'prompt' column contains the fully formatted input
    # including any instructions (like asking for YAML).
    # Example: If you need templating: prompt_template="{{prompt}}" (usually default)
)

# --- Start GRPO Training Job ---
print(f"Starting GRPO training job for dataset '{PREDIBASE_DATASET_NAME}'...")
print(f"Base Model: {BASE_MODEL}")
print(f"Adapter will be saved to repository: '{PREDIBASE_REPO_NAME}'")
print(f"Config: {grpo_config}")

try:
    # This submits the job to Predibase's managed infrastructure
    adapter = pb.adapters.create(
        config=grpo_config,
        dataset=PREDIBASE_DATASET_NAME,
        repo=PREDIBASE_REPO_NAME,
        description=f"GRPO fine-tuning for Cluedo memory update task. Base: {BASE_MODEL}",
    )
    print("\n--- GRPO Job Submitted Successfully! ---")
    print(f"Adapter Name: {adapter.name}")
    print(f"Adapter Version: {adapter.version}")
    print(f"Predibase Repository: {adapter.repo}")
    print("\nMonitor the job progress in the Predibase UI.")
    print(f"Once completed, the trained adapter can be deployed from: {adapter.repo}/{adapter.version}")

except Exception as e:
    print(f"\n--- Error submitting GRPO job ---")
    print(e)
