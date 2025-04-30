import art
import asyncio
import json
import random
from pathlib import Path
from typing import Any, Iterator, TypedDict, Optional
import yaml  # <-- Add YAML import

from dotenv import load_dotenv
import openai # ART uses openai client interface

# Load environment variables (like OPENAI_API_KEY, potentially needed by ART)
load_dotenv()

# --- Reward Functions (Copied and adapted from clue-tiny-grpo/train.py) ---

# Helper function to calculate reward for memory updates
def calculate_memory_update_reward(completion_text: str, ground_truth_deductions: Optional[list[str]]) -> float:
    """Calculates reward based on correctness of deduced cards in memory updates (expects YAML)."""
    try:
        # Attempt to parse the completion as YAML
        completion_yaml = yaml.safe_load(completion_text)
        if not isinstance(completion_yaml, dict):
             # Penalize if YAML is valid but not a dictionary
             # print(f"YAML content is not a dictionary: {completion_text[:100]}...")
             return 0.0 
        
        # Key updated to match llm.js convention
        predicted_deductions = set(completion_yaml.get("newlyDeducedCards", [])) 
        truth_set = set(ground_truth_deductions if ground_truth_deductions else [])

        if not predicted_deductions and not truth_set:
            return 1.0 # Correctly deduced nothing new when nothing was expected

        intersection = len(predicted_deductions.intersection(truth_set))
        # Use precision: reward based on how many of the *predicted* deductions were correct
        precision = intersection / len(predicted_deductions) if len(predicted_deductions) > 0 else 0.0
        # Use recall: reward based on how many of the *ground truth* deductions were found
        recall = intersection / len(truth_set) if len(truth_set) > 0 else 0.0
        
        # F1 Score might be a good balance
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Give a full reward only if perfect match
        if len(predicted_deductions) == len(truth_set) and intersection == len(truth_set):
             return 1.0
        # Use F1 score as reward otherwise
        elif (precision + recall) > 0:
             return f1
        # If prediction is empty but truth is not, reward is 0
        elif not predicted_deductions and truth_set:
             return 0.0
        # If prediction is not empty but truth is, reward is 0 (penalize hallucination)
        elif predicted_deductions and not truth_set:
             return 0.0
        else: # Both empty already handled, this case shouldn't be hit
             return 0.0

    except yaml.YAMLError:
        # Penalize non-YAML output for memory updates
        # print(f"YAMLError calculating reward for: {completion_text[:100]}...")
        return 0.0 # Strict penalty for invalid YAML
    except Exception as e:
        print(f"Warning: Error calculating memory update reward: {e}")
        return 0.0

# Helper function for basic reward (e.g., suggestion/accusation format check)
def calculate_basic_reward(completion_text: str) -> float:
    """Calculates a basic reward, e.g., for valid YAML format in suggestions/accusations."""
    try:
        content = yaml.safe_load(completion_text)
        # Basic reward for outputting valid YAML (and being a dict/list perhaps)
        if isinstance(content, (dict, list)):
            return 0.2 # Small reward for valid YAML structure
        else:
             # print(f"YAML content is not dict/list: {completion_text[:100]}...")
             return 0.05 # Very small reward for valid YAML but wrong type
    except yaml.YAMLError:
        # print(f"YAMLError calculating basic reward for: {completion_text[:100]}...")
        return 0.0 # Penalize invalid YAML

# --- Data Loading ---

# Define structure for type hinting
class ClueInteraction(TypedDict):
    interaction_type: str
    prompt: str
    ground_truth_deductions: Optional[list[str]]
    # Add other fields if they exist in the jsonl and are needed
    # e.g., game_state: dict

def read_jsonl(file_name: str | Path) -> Iterator[dict]:
    """Reads a JSONL file line by line."""
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {file_name}: {line[:100]}...")


def load_clue_data(file_path: str | Path, max_rows: Optional[int] = None) -> list[ClueInteraction]:
    """Loads Clue interaction data from the JSONL file."""
    data = []
    # Construct the relative path from ART/examples/ to clue-tiny-grpo/data/
    # Assuming clue-tiny-grpo is a sibling directory to ART
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent # Goes up two levels from ART/examples to workspace root
    data_file = base_dir / "clue-tiny-grpo" / "data" / "cluedo_interactions.jsonl"
    
    print(f"Attempting to load data from: {data_file}")
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        # Try the path provided directly, relative to workspace?
        data_file = Path(file_path) 
        print(f"Trying alternative path: {data_file}")
        if not data_file.exists():
             print(f"Error: Data file not found at alternative path either. Please check the path.")
             return []

    count = 0
    for item in read_jsonl(data_file):
        # Adapt this part if the structure of your JSONL is different
        interaction = ClueInteraction(
            interaction_type=item.get("interaction_type", "unknown"),
            prompt=item.get("prompt", ""),
            ground_truth_deductions=item.get("ground_truth_deductions") # Can be None
            # Add other fields as needed
        )
        if interaction["prompt"]: # Only add if prompt is not empty
            data.append(interaction)
            count += 1
            if max_rows is not None and count >= max_rows:
                break
        else:
             print(f"Skipping interaction due to empty prompt: {item}")
             
    print(f"Loaded {len(data)} Clue interaction examples.")
    return data

# --- ART Model Definition ---

# Define the trainable model using art.TrainableModel
# Customize base_model, project, name as needed
model = art.TrainableModel(
    name="clue-agent-001", # A name for this specific run/agent
    project="cluedo-art-training", # Project name for organization
    # Choose a base model compatible with ART's local API (e.g., from Hugging Face)
    # Using the same small model as clue-tiny-grpo for comparison
    base_model="Qwen/Qwen3-4B", 
    # Add any necessary config, e.g., GPU utilization if needed
    # _internal_config={"init_args": {"gpu_memory_utilization": 0.8}},
)

# --- Clue Rollout Function ---

async def clue_rollout(
    client: openai.AsyncOpenAI, 
    interaction: ClueInteraction,
    max_tokens: int = 2000, # Max tokens to generate for the response
    temperature: float = 0.7,
    top_p: float = 0.9
) -> art.Trajectory:
    """Performs a single rollout for a Clue interaction using the ART model."""
    
    prompt_text = interaction["prompt"]
    interaction_type = interaction["interaction_type"]
    ground_truth = interaction["ground_truth_deductions"]

    # Format prompt (similar to clue-tiny-grpo, ensure model knows to output YAML)
    # This might need adjustment based on the base model's instruction following capabilities
    if interaction_type == "memory_update":
        # Updated instruction asking for YAML
        formatted_prompt = prompt_text + "\n\nIMPORTANT: Respond ONLY with a YAML object containing the key 'newlyDeducedCards' as a list. Example:\nnewlyDeducedCards:\n  - Card1\n  - Card2"
    elif interaction_type in ["suggestion", "accusation"]:
        # Assume these also require YAML? Adjust prompt accordingly.
         formatted_prompt = prompt_text + "\n\nIMPORTANT: Respond ONLY with a valid YAML object representing your move."
    else:
        formatted_prompt = prompt_text # For other types, maybe no strict YAML needed?

    messages: art.Messages = [{"role": "user", "content": formatted_prompt}]

    try:
        chat_completion = await client.chat.completions.create(
            messages=messages, 
            model=model.name, # Use the ART model name
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # Ensure the response format is text if not using JSON mode explicitly
            # response_format={"type": "text"},
        )
        choice = chat_completion.choices[0]
        completion_text = choice.message.content
        
        if not isinstance(completion_text, str):
             completion_text = "" # Handle potential None or other types

        # --- DEBUG: Print model output ---
        print(f"--- Interaction Type: {interaction_type} ---")
        print(f"Model Completion: {completion_text}")
        print(f"Ground Truth (Memory): {ground_truth}")
        # --- END DEBUG ---

    except Exception as e:
        print(f"Error during model generation: {e}")
        completion_text = "" # Assign empty string on error
        # Create a dummy choice to avoid downstream errors
        choice = openai.types.chat.chat_completion.ChatCompletion.Choice(
             finish_reason="error", 
             index=0, 
             message=openai.types.chat.chat_completion_message.ChatCompletionMessage(role="assistant", content=""),
             logprobs=None
        )

    # Calculate reward based on interaction type
    reward = 0.0
    metrics = {}
    if interaction_type == "memory_update":
        reward = calculate_memory_update_reward(completion_text, ground_truth)
        metrics["memory_reward"] = reward
    else:
        # Use basic reward (e.g., JSON validity) for suggestions, accusations, etc.
        reward = calculate_basic_reward(completion_text)
        metrics["basic_reward"] = reward
        
    # Add accuracy metric (e.g., 1 if reward is > threshold, 0 otherwise)
    # Simple accuracy: 1 if reward is max (1.0 for memory, 0.2 for basic)
    is_accurate = False
    if interaction_type == "memory_update" and reward == 1.0:
         is_accurate = True
    elif interaction_type != "memory_update" and reward == 0.2:
         is_accurate = True
         
    metrics["accuracy"] = 1.0 if is_accurate else 0.0
    metrics["reward"] = reward # Include raw reward in metrics too

    return art.Trajectory(
        # Ensure messages_and_choices includes the assistant's message
        messages_and_choices=[*messages, choice], 
        reward=reward, 
        metrics=metrics
    )


# --- Main Execution Logic (Placeholder) ---
async def main():
    print("Starting Clue training script using ART...")
    
    # Register the model with the local API before use
    await model.register(art.LocalAPI()) # Use LocalAPI for local training
    print(f"ART Model '{model.name}' registered with LocalAPI.")

    # 1. Load Data
    # Adjust path if needed, this assumes running from workspace root or ART/examples
    # Use the relative path, load_clue_data will resolve it
    clue_data = load_clue_data("clue-tiny-grpo/data/cluedo_interactions.jsonl", max_rows=500) # Load subset for faster testing
    if not clue_data:
        return

    # Filter data to only include memory_update interactions
    memory_update_data = [item for item in clue_data if item["interaction_type"] == "memory_update"]
    print(f"Filtered data to {len(memory_update_data)} memory_update interactions.")
    if not memory_update_data:
        print("Error: No memory_update interactions found in the data. Cannot train.")
        return

    # Example: Split data (simple split)
    random.seed(42)
    random.shuffle(memory_update_data) # Shuffle the filtered data
    split_idx = int(len(memory_update_data) * 0.9)
    train_data = memory_update_data[:split_idx] # Use filtered data for train
    val_data = memory_update_data[split_idx:]   # Use filtered data for val
    print(f"Train size: {len(train_data)}, Validation size: {len(val_data)}")

    # 2. Get OpenAI Client from ART Model
    openai_client = model.openai_client()

    # --- Training Loop ---
    num_training_steps = 100 # Total number of training steps
    rollouts_per_step = 8  # Number of interactions to collect rollouts for per step
    group_size = 4 # Number of rollouts per interaction (like in clue-tiny-grpo)
    val_interval = 10 # How often to run validation

    # Training config for model.train()
    # We might need to adjust the learning rate
    train_config = art.TrainConfig(learning_rate=1e-5) 

    print(f"Starting training for {num_training_steps} steps...")
    # Resume from the last step if checkpoint exists
    start_step = await model.get_step() 
    print(f"Resuming from step {start_step}")

    for i in range(start_step, num_training_steps):
        print(f"--- Step {i+1}/{num_training_steps} ---")

        # --- Gather Training Trajectories ---
        # Select a batch of training interactions
        # Simple slicing for now, could use DataLoader for more robust sampling
        start_idx = (i * rollouts_per_step) % len(train_data)
        end_idx = start_idx + rollouts_per_step
        train_batch = (train_data * ( (i*rollouts_per_step // len(train_data)) + 2) )[start_idx:end_idx] # Wrap around data
        
        print(f"Gathering {group_size} rollouts for {len(train_batch)} training interactions...")
        train_groups = await art.gather_trajectory_groups(
            (
                # For each interaction, create a group of `group_size` rollouts
                art.TrajectoryGroup(clue_rollout(openai_client, interaction) for _ in range(group_size))
                for interaction in train_batch
            ),
            pbar_desc=f"Train Step {i+1}",
        )
        print(f"Gathered {len(train_groups)} training trajectory groups.")

        # --- Training Step ---
        if train_groups:
            print(f"Training model...")
            # Call ART's built-in train function
            await model.train(
                train_groups,
                config=train_config,
            )
            print(f"Training step {i+1} completed.")
            # Optionally save checkpoint after training step
            # await model.save_checkpoint() # Removed incorrect call
            await model.delete_checkpoints() # Keep last 2 checkpoints
        else:
            print("No training groups gathered, skipping training step.")

        # --- Validation Step (Periodically) ---
        if (i + 1) % val_interval == 0 and val_data:
            print(f"Running validation...")
            # Gather validation trajectories (usually fewer rollouts per interaction needed)
            val_groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(clue_rollout(openai_client, interaction) for _ in range(1)) # Only 1 rollout for val
                    for interaction in val_data # Use the whole validation set
                ),
                pbar_desc=f"Validation Step {i+1}",
            )
            print(f"Gathered {len(val_groups)} validation trajectory groups.")
            
            # Log validation metrics (ART handles aggregation)
            if val_groups:
                await model.log(val_groups)
                print("Validation metrics logged.")
            else:
                 print("No validation groups gathered.")

    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(main()) 