from collections.abc import Callable
import json
from pathlib import Path
import random
import re
from typing import Any, Iterator, Optional
import os # Add os import
import yaml # <-- Add YAML import

# Make wandb import optional
try:
    import wandb
    wandb_available = True
except (ImportError, AttributeError) as e:
    print(f"WandB import failed: {e}. Logging will be disabled.")
    wandb_available = False

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)
from loss import approx_kl_divergence, GRPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    device_map=None,
) -> tuple[LlamaForCausalLM, PreTrainedTokenizer]:
    
    # Get token from environment variable
    token = os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("Warning: HUGGING_FACE_HUB_TOKEN environment variable not set. Access to gated models may fail.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token) # Pass token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Determine optimal dtype and if flash attn can be used based on GPU capability
    attn_implementation = None
    torch_dtype = "auto" # Default
    quantization_config = None # Default: no quantization

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        print(f"Detected CUDA capability: {capability}")
        
        # Configure BitsAndBytes 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, # Compute in float16 on T4
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype = torch.float16 # Quantization needs a compute dtype
        print("Enabled 4-bit quantization (bnb_4bit_compute_dtype=float16).")

        if capability >= (8, 0): # Ampere and newer
            print("GPU capability >= 8.0. Checking for Flash Attention.")
            # bnb_4bit_compute_dtype could be bfloat16 here, but float16 is safer for wider compatibility
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                print("Flash Attention is available, enabling it.")
            except ImportError:
                print("Flash Attention import failed, using standard attention.")
        else: # Older GPUs (like T4)
            print("GPU capability < 8.0. Using standard attention.")
            print("Flash Attention not supported or optimal on this GPU.")
    else:
        print("CUDA not available, using CPU (Quantization disabled).")
        quantization_config = None # Disable quantization on CPU
        torch_dtype = "auto"

    # Model loading parameters
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "quantization_config": quantization_config, # Pass quantization config
    }
    
    # Only add flash attention if supported and imported
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    
    print(f"Loading model with args: {model_kwargs}")
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        token=token, # Pass token here too
        **model_kwargs
    )
    
    # If using quantization and device_map='auto', model is already on GPU.
    # If not using device_map='auto', might need model.to(device)
    
    # Prepare model for k-bit training if quantized
    if quantization_config and hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        from peft import prepare_model_for_kbit_training
        print("Preparing quantized model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        # Note: Further PEFT/LoRA setup would be needed here for actual fine-tuning
        # For GRPO, we might not need LoRA if only the reference model is quantized,
        # but if the main 'model' is also quantized, LoRA might be necessary.
        # Let's assume for now only reference_model uses quantization for rollouts.
        # Revisit this if the main 'model' forward pass causes issues.

    return model, tokenizer


# Cluedo System Prompt (Example - adjust as needed)
cluedo_system_prompt = """\"\"\"You are an AI assistant playing the game of Cluedo. Your goal is to deduce the murderer, weapon, and room by making suggestions, evaluating challenges, and updating your memory based on game events. Respond accurately and strategically based on the provided information. Respond ONLY with the requested JSON format.\"\"\"
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizer,
    task_data: dict, # Now expects the dictionary loaded from JSONL
    num_rollouts: int,
    max_length: int = 768,  # Reduced max_length
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:

    model.eval()
    
    # Extract data from this task, handling the format from the custom collate function
    # With batch_size=1 and custom_collate, we get {key: [value]} instead of just {key: value}
    # Get the first (and only) item from each list if it exists
    if isinstance(task_data, dict):
        interaction_type_val = task_data.get("interaction_type", ["unknown"])
        prompt_text_val = task_data.get("prompt", [None])
        ground_truth_val = task_data.get("ground_truth_deductions", [None])
        
        # Check if we have lists and extract the first item if they're not empty
        interaction_type = interaction_type_val[0] if isinstance(interaction_type_val, list) and interaction_type_val else "unknown"
        prompt_text = prompt_text_val[0] if isinstance(prompt_text_val, list) and prompt_text_val else None
        ground_truth_deductions = ground_truth_val[0] if isinstance(ground_truth_val, list) and ground_truth_val else None
    else:
        # If somehow task_data is not a dict, provide safe defaults
        print(f"Warning: task_data is not a dictionary: {type(task_data)}")
        interaction_type = "unknown"
        prompt_text = None
        ground_truth_deductions = None

    if not prompt_text:
        print("Warning: Skipping rollout due to missing prompt text.")
        # Return empty tensors with correct sizes to maintain expected shapes
        # Use device from model for consistency
        device = model.device
        # Create dummy tensors with proper batch dimension
        dummy_seq = torch.zeros((num_rollouts, 1), dtype=torch.long, device=device)
        dummy_returns = torch.zeros((num_rollouts, 1), dtype=torch.float, device=device)
        dummy_mask = torch.zeros((num_rollouts, 1), dtype=torch.bool, device=device)
        return dummy_seq, dummy_returns, dummy_mask, [""] * num_rollouts

    # 1. format prompt with YAML formatting instructions
    if interaction_type == "memory_update":
        # Add YAML formatting instruction
        chat_prompt = prompt_text + "\n\nIMPORTANT: Respond ONLY with a YAML object containing the key 'newlyDeducedCards' as a list. Example:\nnewlyDeducedCards:\n  - Card1\n  - Card2"
    else:
        # For other interaction types, request generic YAML (though we filtered these out)
        chat_prompt = prompt_text + "\n\nIMPORTANT: Respond ONLY with a valid YAML object."

    # Efficient batched tokenization for GPU
    try:
        model_inputs = tokenizer(
            [chat_prompt] * num_rollouts, # Repeat prompt for batch generation
            return_tensors="pt",
            padding=True,
            padding_side="left", # Important for generation
            truncation=True,
            max_length=max_length - 100, # Ensure space for generation
            return_attention_mask=True,
        ).to(model.device)
    except Exception as e:
        print(f"Error during tokenization: {e}")
        # Return properly sized dummy tensors
        device = model.device
        dummy_seq = torch.zeros((num_rollouts, 1), dtype=torch.long, device=device)
        dummy_returns = torch.zeros((num_rollouts, 1), dtype=torch.float, device=device)
        dummy_mask = torch.zeros((num_rollouts, 1), dtype=torch.bool, device=device)
        return dummy_seq, dummy_returns, dummy_mask, [""] * num_rollouts

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    
    # Ensure input tensors have the expected batch size
    if input_ids.shape[0] != num_rollouts:
        print(f"Warning: Input tensor batch size mismatch. Expected {num_rollouts}, got {input_ids.shape[0]}. Adjusting...")
        # Either expand or contract to match expected batch size
        if input_ids.shape[0] < num_rollouts:
            # Repeat tensors to achieve desired batch size
            repeats_needed = (num_rollouts + input_ids.shape[0] - 1) // input_ids.shape[0]
            input_ids = input_ids.repeat(repeats_needed, 1)[:num_rollouts]
            attention_mask = attention_mask.repeat(repeats_needed, 1)[:num_rollouts]
        else:
            # Truncate to expected batch size
            input_ids = input_ids[:num_rollouts]
            attention_mask = attention_mask[:num_rollouts]

    # 2. sample completions with efficient generation config
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=64, # Reduced max_new_tokens
        pad_token_id=pad_token_id,
        do_stream=False, # Disable streaming for batch efficiency
    )
    
    # Use efficient generation
    try:
        # Try using mixed precision on GPU - use float16 for T4 compatibility
        with torch.cuda.amp.autocast(dtype=torch.float16): 
            sequence_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
    except RuntimeError as e:
        # Fall back to standard precision if autocast fails
        print(f"Warning: Mixed precision generation failed, falling back to standard precision: {e}")
        try:
            sequence_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
        except Exception as gen_e:
            print(f"Error during generation: {gen_e}")
            # Return properly sized dummy tensors
            device = model.device
            dummy_seq = torch.zeros((num_rollouts, 1), dtype=torch.long, device=device)
            dummy_returns = torch.zeros((num_rollouts, 1), dtype=torch.float, device=device)
            dummy_mask = torch.zeros((num_rollouts, 1), dtype=torch.bool, device=device)
            return dummy_seq, dummy_returns, dummy_mask, [""] * num_rollouts
    
    # --- Check Output Shape --- 
    if sequence_ids.shape[0] != num_rollouts:
        print(f"[Rollout Error] Generated batch size mismatch! Expected {num_rollouts}, Got {sequence_ids.shape[0]}. Adjusting tensors...")
        # Adjust tensor to have the expected batch size
        if sequence_ids.shape[0] < num_rollouts:
            # Repeat sequences to match the expected batch size
            repeats_needed = (num_rollouts + sequence_ids.shape[0] - 1) // sequence_ids.shape[0]
            sequence_ids = sequence_ids.repeat(repeats_needed, 1)[:num_rollouts]
        else:
            # Truncate to expected batch size
            sequence_ids = sequence_ids[:num_rollouts]
    # --- End Check --- 
    
    try:
        completions = tokenizer.batch_decode(
            sequence_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )
        
        # Ensure completions list has the correct length
        if len(completions) != num_rollouts:
            print(f"Warning: Completions list has incorrect length. Expected {num_rollouts}, got {len(completions)}. Adjusting...")
            if len(completions) < num_rollouts:
                # Pad with empty strings
                completions.extend([""] * (num_rollouts - len(completions)))
            else:
                # Truncate
                completions = completions[:num_rollouts]
    except Exception as e:
        print(f"Error during completion decoding: {e}")
        completions = [""] * num_rollouts  # Empty completions
    
    # Create action mask (masking prompt tokens, keeping completion tokens)
    try:
        action_mask = torch.ones_like(sequence_ids, dtype=torch.bool)
        action_mask[:, :input_ids.shape[1]] = False # Mask prompt tokens
        # Mask padding tokens in the completion part
        completion_start_index = input_ids.shape[1]
        for i in range(sequence_ids.shape[0]):
            # Find the first pad token *after* the prompt
            pads = (sequence_ids[i, completion_start_index:] == pad_token_id).nonzero()
            if len(pads) > 0:
                first_pad_index = pads[0].item() + completion_start_index
                action_mask[i, first_pad_index:] = False
    
        # Align with log_probs (logits[:, :-1])
        # Check shapes before slicing
        if action_mask.shape[1] > 1:
            action_mask = action_mask[:, 1:] 
        else:
            # Handle case where action_mask only has one column
            print("Warning: action_mask has only one column, creating new mask of proper size")
            action_mask = torch.zeros((sequence_ids.shape[0], sequence_ids.shape[1] - 1), dtype=torch.bool, device=sequence_ids.device)
    except Exception as e:
        print(f"Error creating action mask: {e}")
        # Create a dummy mask of the expected shape
        action_mask = torch.zeros((num_rollouts, sequence_ids.shape[1] - 1), dtype=torch.bool, device=sequence_ids.device)

    # --- DEBUG --- Print shapes before returning from rollout
    print(f"  [Rollout Debug] sequence_ids shape: {sequence_ids.shape}")
    print(f"  [Rollout Debug] action_mask shape: {action_mask.shape}")
    
    # Verify tensor shapes are compatible
    if sequence_ids.shape[1] - 1 != action_mask.shape[1]:
        print(f"  [Rollout Debug] WARNING: Shape mismatch between sequence_ids and action_mask!")
        # Fix the action_mask shape
        expected_cols = sequence_ids.shape[1] - 1
        if action_mask.shape[1] < expected_cols:
            # Pad with False
            padding = torch.zeros((action_mask.shape[0], expected_cols - action_mask.shape[1]), 
                                 dtype=torch.bool, device=action_mask.device)
            action_mask = torch.cat([action_mask, padding], dim=1)
        else:
            # Truncate
            action_mask = action_mask[:, :expected_cols]
    # --- END DEBUG ---

    # 3. determine rewards based on interaction type
    try:
        returns = torch.zeros(num_rollouts, 1, dtype=torch.float, device=sequence_ids.device)
        for i, completion in enumerate(completions):
            reward = 0.0
            if interaction_type == "memory_update":
                # Use ground truth deductions for reward calculation
                reward = calculate_memory_update_reward(completion, ground_truth_deductions)
            else:
                # Use basic reward (e.g., JSON validity) for other types
                reward = calculate_basic_reward(completion)
    
            returns[i] = reward
    except Exception as e:
        print(f"Error calculating rewards: {e}")
        # Create a new returns tensor with proper size
        returns = torch.zeros(num_rollouts, 1, dtype=torch.float, device=sequence_ids.device)

    # Final verification of shapes
    print(f"  [Rollout Final] Returning shapes - sequence_ids: {sequence_ids.shape}, returns: {returns.shape}, action_mask: {action_mask.shape}")
    return sequence_ids, returns, action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    try:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
        output = model.forward(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = output["logits"]
        
        # Debug shapes to help identify potential mismatches
        if logits.shape[1] - 1 != sequence_ids.shape[1] - 1:
            print(f"Warning: Shape mismatch! logits[:, :-1]: {logits[:, :-1].shape}, sequence_ids[:, 1:]: {sequence_ids[:, 1:].shape}")
        
        log_probs = sequence_log_probs_from_logits(
            logits=logits[:, :-1].to(torch.float32),
            output_ids=sequence_ids[:, 1:],
        )
        return log_probs
    except RuntimeError as e:
        print(f"Error in sequences_log_probs: {e}")
        print(f"Shapes: sequence_ids={sequence_ids.shape}, attention_mask={attention_mask.shape}, logits={logits.shape if 'logits' in locals() else 'N/A'}")
        raise


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


# Helper function to calculate reward for memory updates
def calculate_memory_update_reward(completion_text: str, ground_truth_deductions: list[str]) -> float:
    try:
        # Attempt to parse the completion as YAML
        completion_yaml = yaml.safe_load(completion_text)
        if not isinstance(completion_yaml, dict):
            # Penalize if YAML is valid but not a dictionary
            # print(f"YAML content is not a dictionary: {completion_text[:100]}...")
            return 0.0
            
        # Adjust key based on expected LLM output format (YAML) for memory update
        predicted_deductions = set(completion_yaml.get("newlyDeducedCards", []))
        truth_set = set(ground_truth_deductions if ground_truth_deductions else [])

        if not predicted_deductions and not truth_set:
            return 1.0 # Correctly deduced nothing new when nothing was expected

        intersection = len(predicted_deductions.intersection(truth_set))
        # Using F1 score as reward metric (similar to ART script)
        precision = intersection / len(predicted_deductions) if len(predicted_deductions) > 0 else 0.0
        recall = intersection / len(truth_set) if len(truth_set) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Give full reward only if perfect match
        if len(predicted_deductions) == len(truth_set) and intersection == len(truth_set):
             return 1.0
        elif (precision + recall) > 0:
             return f1
        elif not predicted_deductions and truth_set:
             return 0.0
        elif predicted_deductions and not truth_set:
             return 0.0
        else:
             return 0.0

    except yaml.YAMLError:
        # print(f"YAMLError calculating reward for: {completion_text[:100]}...")
        return 0.0 # Penalize invalid YAML
    except Exception as e:
        print(f"Warning: Error calculating reward: {e}")
        return 0.0


# Helper function for basic reward (e.g., suggestion/accusation format check)
def calculate_basic_reward(completion_text: str) -> float:
     try:
        content = yaml.safe_load(completion_text)
        # Basic reward for outputting valid YAML (and being a dict/list)
        if isinstance(content, (dict, list)):
            return 0.2 # Use 0.2 to match ART script's basic reward
        else:
             return 0.05 # Small reward for valid YAML but wrong type
     except yaml.YAMLError:
         return 0.0 # Penalize invalid YAML


def custom_collate(batch):
    """Custom collate function that can handle None values in the dataset"""
    if not batch:
        return {}
    
    # If we get a list of dictionaries, convert to a dictionary of lists
    if isinstance(batch[0], dict):
        result = {}
        # For each key in the first dictionary
        for key in batch[0].keys():
            # Collect all values for this key across all dictionaries
            values = [d.get(key) for d in batch]
            # Store in the result dictionary
            result[key] = values
        return result
    
    # For non-dictionary batches, just return as is
    return batch


def main():
    # Declare that we are using the global wandb_available variable
    global wandb_available 
    
    seed = 42
    # Re-enable wandb logging
    wandb_project = "cluedo_grpo"  # Set WandB project name
    device_index = 0
    # Using 1B model for faster loading and less memory consumption
    model_name = "Qwen/Qwen3-4B"  # Switched to 4B model for better results
    checkpoint_path = Path("./output_cluedo")  # Separate output dir
    checkpoint_interval = 20
    train_batch_size = 8  # Reduced for easier startup
    lr = 5e-6  # Starting learning rate
    kl_weight = 0.01  # D_KL coefficient in loss
    clip_eps = 0.2  # PPO clipping epsilon

    group_size = 4  # Reduced further
    rollouts_per_step = 16  # Kept same for now
    epochs_per_step = 1  
    max_norm = 1.0  

    # rollout params
    max_length = 768 # Reduced max_length (matches rollout default)
    top_p = 0.9  
    temperature = 0.7  

    # Prioritize CUDA GPU
    if torch.cuda.is_available():
        device = torch.device("cuda", device_index)
        print(f"Using CUDA device: {device_index}")
        print(f"GPU: {torch.cuda.get_device_name(device_index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device_index).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU. Training will be much slower.")
    
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # Load models - use 4-bit quantization for reference model potentially
    print(f"Loading models: {model_name}")
    # Load reference model with quantization potentially enabled by load_model function
    reference_model, _ = load_model(model_name, device_map="auto") 
    # Load main training model - might disable quantization here if training full model
    # For now, let's load both the same way. Revisit if training pass fails.
    model, tokenizer = load_model(model_name, device_map="auto")
    print("Models loaded successfully")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Ensure pad token is set for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.pad_token_id

    # Load Cluedo data
    print("Loading Cluedo interaction data...")
    # Use a path relative to this script's location
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "cluedo_interactions.jsonl"
    print(f"Attempting to load data from: {data_file}")
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}. Please check the path.")
        return
        
    prompts_data = read_jsonl(data_file)
    prompts_list = list(prompts_data) # Load all into memory for DataLoader
    print(f"Loaded {len(prompts_list)} Cluedo interaction examples.")

    if not prompts_list:
        print("Error: No data loaded from the file.")
        return

    # Filter data to only include memory_update interactions
    memory_update_data = [item for item in prompts_list if item.get("interaction_type") == "memory_update"]
    print(f"Filtered data to {len(memory_update_data)} memory_update interactions.")
    if not memory_update_data:
        print("Error: No memory_update interactions found in the data. Cannot train.")
        return

    # Note: DataLoader will yield dictionaries from the filtered list
    prompt_loader = DataLoader(
        memory_update_data, # Use the filtered data
        batch_size=1, # Process one prompt data dict at a time for rollout
        shuffle=True,
        drop_last=True, # Avoid partial batches if rollouts_per_step doesn't divide dataset size
        collate_fn=custom_collate, # Use our custom collate function
    )
    prompt_iterator = iter(prompt_loader) # Make it an iterator

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=clip_eps, kl_weight=kl_weight)

    # Initialize wandb only if available
    if wandb_project is not None and wandb_available:
        try:
            wandb.init(project=wandb_project)
            print(f"WandB initialized with project: {wandb_project}")
        except Exception as e:
            print(f"Failed to initialize WandB: {e}. Continuing without logging.")
            wandb_available = False
    else:
        print("WandB logging disabled.")
        wandb_available = False

    # Create checkpoint dir
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    # Adjust total steps based on dataset size and how many times you want to iterate
    total_training_steps = 500 

    for step in range(total_training_steps):
        print(f"--- Step {step + 1} / {total_training_steps} ---")
        experiences = []
        model.eval() # Set model to eval for rollouts

        # --- Rollout Phase ---
        # Collect rollouts for a number of prompts determined by rollouts_per_step
        prompts_processed_this_step = 0
        rollout_data_collected = []

        while prompts_processed_this_step < rollouts_per_step:
            try:
                # Get next task data dictionary from the loader
                current_task_data = next(prompt_iterator)
                
                # Log the type we're processing
                interaction_type = "unknown"
                if isinstance(current_task_data, dict) and "interaction_type" in current_task_data:
                    if isinstance(current_task_data["interaction_type"], list):
                        if current_task_data["interaction_type"]:
                            interaction_type = current_task_data["interaction_type"][0]
                    else:
                        interaction_type = current_task_data["interaction_type"]
                        
                print(f"Rolling out for prompt type: {interaction_type}...")
                
            except StopIteration:
                # Reset iterator if dataset is exhausted
                print("Resetting prompt data loader.")
                prompt_iterator = iter(prompt_loader)
                try:
                    current_task_data = next(prompt_iterator)
                except StopIteration:
                    print("Error: Dataset is empty or DataLoader is not working correctly.")
                    break
                
                # Get the interaction type after resetting
                interaction_type = "unknown"
                if isinstance(current_task_data, dict) and "interaction_type" in current_task_data:
                    if isinstance(current_task_data["interaction_type"], list):
                        if current_task_data["interaction_type"]:
                            interaction_type = current_task_data["interaction_type"][0]
                    else:
                        interaction_type = current_task_data["interaction_type"]
                print(f"Rolling out for prompt type (after reset): {interaction_type}...")
            
            # Perform rollouts for the current task
            try:
                seq_ids, returns, action_mask, completions = rollout(
                    model=reference_model, # Use reference model for rollouts
                    tokenizer=tokenizer,
                    task_data=current_task_data,
                    num_rollouts=group_size, # Generate 'group_size' completions per prompt
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                # Check if rollout produced valid data (non-empty sequence_ids)
                if seq_ids.numel() == 0 or seq_ids.shape[0] == 0:
                    print("Skipping empty or failed rollout result.")
                    continue
                    
                # Added validation: Check for correct batch dimension before adding to collected data
                if seq_ids.shape[0] != group_size:
                    print(f"Skipping rollout with incorrect batch size: Expected {group_size}, got {seq_ids.shape[0]}")
                    continue
                
                # Validate all tensor shapes match the expected group_size
                if action_mask.shape[0] != group_size or returns.shape[0] != group_size:
                    print(f"Skipping rollout with inconsistent tensor shapes: seq_ids={seq_ids.shape}, action_mask={action_mask.shape}, returns={returns.shape}")
                    continue

                # Calculate advantages for this group of rollouts
                advantages = group_advantages(returns)

                # Get log probs from reference model for KL divergence penalty
                with torch.no_grad():
                    ref_log_probs = sequences_log_probs(
                        reference_model,
                        sequence_ids=seq_ids,
                        attention_mask=(seq_ids != pad_token_id),
                    )
                    ref_log_probs = ref_log_probs.detach()

                # Store data needed for training step
                rollout_data_collected.append(
                    {
                        "sequences": seq_ids.to(cpu_device),
                        "action_mask": action_mask.to(cpu_device),
                        "returns": returns.to(cpu_device),
                        "advantages": advantages.to(cpu_device),
                        "log_probs_ref": ref_log_probs.to(cpu_device),
                    }
                )
                prompts_processed_this_step += 1
                print(f"Rollouts collected for prompt {prompts_processed_this_step}/{rollouts_per_step}")
            except Exception as e:
                print(f"Error during rollout: {e}")
                continue  # Skip to the next prompt


        # --- Training Phase ---
        model.train() # Set model to train
        
        # Skip training if no valid rollout data was collected
        if not rollout_data_collected:
            print("No valid rollout data collected this step. Skipping training.")
            continue
        
        # Filter out any experiences with incorrect shapes before batch joining
        # First, find the most common shape pattern for each key
        shape_counts = {
            "sequences": {},
            "action_mask": {},
            "log_probs_ref": {}
        }
        
        # First pass: count shapes
        for exp_dict in rollout_data_collected:
            for key in shape_counts.keys():
                if key in exp_dict and isinstance(exp_dict[key], torch.Tensor):
                    shape_str = '_'.join(str(dim) for dim in exp_dict[key].shape)
                    shape_counts[key][shape_str] = shape_counts[key].get(shape_str, 0) + 1
        
        # Find dominant shapes
        dominant_shapes = {}
        for key, counts in shape_counts.items():
            if counts:
                dominant_shape_str = max(counts.items(), key=lambda x: x[1])[0]
                dominant_shapes[key] = tuple(int(dim) for dim in dominant_shape_str.split('_'))
                print(f"Dominant shape for {key}: {dominant_shapes[key]} ({counts[dominant_shape_str]}/{len(rollout_data_collected)} experiences)")
        
        # Second pass: filter based on dominant shapes
        filtered_rollout_data = []
        for i, exp_dict in enumerate(rollout_data_collected):
            is_valid = True
            
            # Check all required keys are present
            required_keys = ["sequences", "action_mask", "returns", "advantages", "log_probs_ref"]
            for key in required_keys:
                if key not in exp_dict or exp_dict[key] is None:
                    print(f"Skipping experience {i}: Missing key {key}")
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Check tensor types
            for key in required_keys:
                if not isinstance(exp_dict[key], torch.Tensor):
                    print(f"Skipping experience {i}: {key} is not a tensor")
                    is_valid = False
                    break
            
            if not is_valid:
                continue
                
            # Check exact shapes for variable sequence tensors
            for key in ["sequences", "action_mask", "log_probs_ref"]:
                if key in dominant_shapes:
                    exp_shape = exp_dict[key].shape
                    dominant_shape = dominant_shapes[key]
                    
                    # Check dimensions match exactly - first batch dimension and sequence length
                    if len(exp_shape) != len(dominant_shape) or exp_shape[0] != dominant_shape[0] or exp_shape[1] != dominant_shape[1]:
                        print(f"Skipping experience {i}: {key} shape {exp_shape} does not match dominant shape {dominant_shape}")
                        is_valid = False
                        break
            
            if not is_valid:
                continue
                
            # Check exact shapes for fixed tensors
            for key in ["returns", "advantages"]:
                if key in exp_dict:
                    exp_shape = exp_dict[key].shape
                    if exp_shape != (group_size, 1):
                        print(f"Skipping experience {i}: {key} shape {exp_shape} does not match expected (group_size, 1)")
                        is_valid = False
                        break
            
            if is_valid:
                filtered_rollout_data.append(exp_dict)
                
        print(f"After strict shape filtering: {len(filtered_rollout_data)}/{len(rollout_data_collected)} experiences kept")
        
        # Skip if all experiences were filtered out
        if not filtered_rollout_data:
            print("All experiences filtered out due to inconsistent shapes. Skipping training step.")
            continue
            
        # Combine collected data into batches
        combined_data = join_experience_batch(filtered_rollout_data)

        # Create TensorDataset for training
        try:
            # Ensure all necessary tensors are present before creating the dataset
            required_keys = ["sequences", "action_mask", "returns", "advantages", "log_probs_ref"]
            # Remove action_log_probs from required fields since it's consistently missing
            if not all(k in combined_data and combined_data[k] is not None for k in required_keys):
                missing = [k for k in required_keys if k not in combined_data or combined_data[k] is None]
                print(f"Error: Missing required tensors for TensorDataset: {missing}. Skipping training step.")
                continue # Skip to next step
            
            # Print the shapes before reshaping
            print("Tensor shapes before reshaping:")
            for key, tensor in combined_data.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  {key}: {tensor.shape}")
            
            # Reshape tensors to ensure consistent dimensions
            # First, determine if we have 3D tensors (batch, group_size, seq_len)
            has_3d_tensors = any(tensor.dim() == 3 for tensor in combined_data.values() if isinstance(tensor, torch.Tensor))
            
            try:
                if has_3d_tensors:
                    # Get the batch size and group size from a 3D tensor
                    batch_size = None
                    group_size = None
                    reference_tensor_key = None
                    
                    # First, find a suitable reference tensor with 3D shape
                    for key, tensor in combined_data.items():
                        if isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
                            batch_size, group_size = tensor.shape[0], tensor.shape[1]
                            reference_tensor_key = key
                            print(f"Using '{key}' as reference tensor with shape {tensor.shape}")
                            break
                            
                    if batch_size is None or group_size is None:
                        print("Error: Could not determine batch_size and group_size from any tensor")
                        continue  # Skip to next iteration
                    
                    print(f"Reshaping all tensors to be compatible with batch_size={batch_size}, group_size={group_size}")
                    
                    # Track which keys were successfully reshaped
                    reshaped_keys = []
                    
                    # Reshape all tensors to [batch_size*group_size, ...] format
                    for key in list(combined_data.keys()):  # Create a list to avoid dict size change during iteration
                        if key not in combined_data or combined_data[key] is None:
                            continue
                            
                        tensor = combined_data[key]
                        if not isinstance(tensor, torch.Tensor):
                            print(f"Skipping '{key}' as it's not a tensor")
                            continue
                            
                        try:
                            if tensor.dim() == 3:  # [batch, group_size, seq_len]
                                # Check if this tensor has compatible batch and group dims
                                if tensor.shape[0] != batch_size or tensor.shape[1] != group_size:
                                    print(f"Warning: Tensor '{key}' has incompatible 3D shape {tensor.shape}, expected first dims to be {batch_size, group_size}")
                                    # Try to adjust the tensor shape if possible
                                    if tensor.shape[0] * tensor.shape[1] == batch_size * group_size:
                                        # Same total elements, can reshape
                                        tensor = tensor.reshape(batch_size, group_size, tensor.shape[2])
                                        print(f"  Reshaped to {tensor.shape}")
                                    else:
                                        print(f"  Cannot reshape '{key}', removing from combined data")
                                        combined_data[key] = None
                                        continue
                                
                                # Reshape 3D tensor to 2D [batch*group_size, seq_len]
                                combined_data[key] = tensor.reshape(-1, tensor.size(-1))
                                reshaped_keys.append(key)
                            elif tensor.dim() == 2:
                                if tensor.size(0) == batch_size * group_size:
                                    # Already in the right format [batch*group_size, *]
                                    reshaped_keys.append(key)
                                    pass
                                elif tensor.size(0) == batch_size:
                                    # Need to repeat for each group item [batch, dim] -> [batch*group_size, dim]
                                    try:
                                        combined_data[key] = tensor.repeat_interleave(group_size, dim=0)
                                        reshaped_keys.append(key)
                                    except Exception as e:
                                        print(f"Error repeating tensor '{key}': {e}")
                                        combined_data[key] = None
                                else:
                                    print(f"Warning: Tensor '{key}' has incompatible 2D shape {tensor.shape}")
                                    # Try to reshape if possible
                                    if tensor.size(0) > batch_size * group_size and tensor.size(0) % (batch_size * group_size) == 0:
                                        # Too many elements, can truncate
                                        combined_data[key] = tensor[:batch_size * group_size]
                                        reshaped_keys.append(key)
                                    elif tensor.size(0) < batch_size * group_size and (batch_size * group_size) % tensor.size(0) == 0:
                                        # Too few elements, can repeat
                                        repeat_factor = (batch_size * group_size) // tensor.size(0)
                                        combined_data[key] = tensor.repeat_interleave(repeat_factor, dim=0)
                                        reshaped_keys.append(key)
                                    else:
                                        print(f"  Cannot reshape '{key}', removing from combined data")
                                        combined_data[key] = None
                            elif tensor.dim() == 1:
                                # Handle 1D tensors - most likely need to be expanded
                                try:
                                    # Expand to [batch*group_size, 1]
                                    if tensor.size(0) == 1:
                                        # Single value, repeat for all items
                                        combined_data[key] = tensor.repeat(batch_size * group_size, 1)
                                    elif tensor.size(0) == batch_size:
                                        # One per batch, repeat for group size
                                        expanded = tensor.unsqueeze(1).expand(batch_size, group_size)
                                        combined_data[key] = expanded.reshape(batch_size * group_size, 1)
                                    elif tensor.size(0) == batch_size * group_size:
                                        # Already the right size, just reshape
                                        combined_data[key] = tensor.reshape(batch_size * group_size, 1)
                                    else:
                                        print(f"  Cannot reshape 1D tensor '{key}' with size {tensor.size(0)}")
                                        combined_data[key] = None
                                    reshaped_keys.append(key)
                                except Exception as e:
                                    print(f"Error reshaping 1D tensor '{key}': {e}")
                                    combined_data[key] = None
                            else:
                                print(f"Warning: Tensor '{key}' has {tensor.dim()} dimensions, not sure how to handle")
                                combined_data[key] = None
                        except Exception as e:
                            print(f"Error processing tensor '{key}': {e}")
                            combined_data[key] = None
                    
                    # Print which keys were successfully reshaped
                    print(f"Successfully reshaped tensors: {reshaped_keys}")
                    
                    # Verify all required tensors were reshaped properly
                    for key in required_keys:
                        if key not in reshaped_keys or combined_data[key] is None:
                            print(f"Warning: Required key '{key}' was not properly reshaped")
            
            except Exception as e:
                print(f"Error during tensor reshaping: {e}")
                continue  # Skip to next step
            
            # Print the shapes after reshaping
            print("Tensor shapes after reshaping:")
            for key, tensor in combined_data.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"  {key}: {tensor.shape}")
                
            # Order matters for TensorDataset - match the order expected by the training loop later
            training_dataset = torch.utils.data.TensorDataset(
                combined_data["sequences"],
                combined_data["action_mask"],
                combined_data["returns"],
                combined_data["advantages"],
                combined_data["log_probs_ref"]
            )
        except Exception as e:
            print(f"Error creating TensorDataset: {e}. Skipping training step.")
            # Print shapes for debugging
            for key, tensor in combined_data.items():
                print(f"  {key}: {tensor.shape if isinstance(tensor, torch.Tensor) else type(tensor)}")
            continue

        # Create DataLoader for training steps
        experience_loader = DataLoader(
            training_dataset, # Use the TensorDataset
            batch_size=train_batch_size, # Actual training batch size
            shuffle=True,
        )

        for _ in range(epochs_per_step): # Train for specified epochs on collected data
            # Unpack the tensors from the dataset based on the order defined above
            for sequences_batch, action_mask_batch, returns_batch, advantages_batch, log_probs_ref_batch in experience_loader:
                optimizer.zero_grad()

                # Move batch to training device
                seq_ids = sequences_batch.to(device)
                action_mask = action_mask_batch.to(device)
                returns = returns_batch.to(device)
                advantages = advantages_batch.to(device)
                ref_log_probs = log_probs_ref_batch.to(device)

                # Calculate current log probs using the trainable model
                attention_mask = (seq_ids != pad_token_id).to(device)
                log_probs = sequences_log_probs(
                    model, seq_ids, attention_mask=attention_mask
                )

                # Calculate loss using the GRPO objective
                loss, stats = objective(
                    log_probs=log_probs,
                    old_log_probs=ref_log_probs,
                    advantages=advantages,
                    returns=returns,
                    action_mask=action_mask,
                )

                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=max_norm) # Clip gradients
                optimizer.step()
                global_step += 1

                # Log stats (optional)
                if wandb_project is not None and wandb_available:
                    wandb.log({**stats, "loss": loss.item()}, step=global_step)
                print(f"Step: {global_step}, Loss: {loss.item():.4f}, Reward Mean: {stats.get('reward/mean', 0.0):.3f}, KL: {stats.get('kl_div', 0.0):.4f}")


        # Save checkpoint periodically
        if (step + 1) % checkpoint_interval == 0:
            output_dir = checkpoint_path / f"checkpoint-{step + 1}"
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Checkpoint saved to {output_dir}")

    print("Training finished.")
    # Save final model
    output_dir = checkpoint_path / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final model saved to {output_dir}")


if __name__ == "__main__":
    main()
