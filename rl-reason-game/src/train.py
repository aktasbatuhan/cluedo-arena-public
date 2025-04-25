#!/usr/bin/env python3

import os
import argparse
from typing import Dict, Any, List, Optional, Callable, Tuple
import random
import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

from dataloader import DataLoader
from game_registry import GameRegistry
from reward_registry import RewardRegistry

# Explicitly import our implementations to ensure they're registered
from games.cluedo import CluedoGame
from rewards.cluedo_reward import CluedoRewardModel

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train an agent using GRPO.")
    parser.add_argument(
        "--game", 
        type=str, 
        default="tictactoe",
        help="Name of the game to train on"
    )
    parser.add_argument(
        "--reward", 
        type=str, 
        default="win_loss",
        help="Name of the reward model to use"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/games",
        help="Directory containing game data"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Base LLM to fine-tune"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="models/grpo-agent",
        help="Directory to save the trained model"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=-1,
        help="Maximum number of training steps (-1 for full epochs)"
    )
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_game_dataset(
    game_name: str,
    data_dir: str,
    tokenizer
) -> Tuple[Dataset, Dict[str, List[str]]]:
    """
    Prepare a dataset from game data for training, returning the dataset
    and a lookup map from prompt to ground truth deductions.
    
    Args:
        game_name: Name of the game
        data_dir: Directory containing game data
        tokenizer: Tokenizer for the model
        
    Returns:
        Tuple containing:
            - Dataset formatted for GRPO training (prompt, completion, reward)
            - Dictionary mapping prompt strings to ground_truth_deductions lists
    """
    # Initialize game with empty config
    game = GameRegistry.get_game(game_name, {})
    
    # Load game data
    data_loader = DataLoader(data_dir)
    # game_data_list should be a list with one entry containing "board_states"
    # as per the load_jsonl_data modification in CluedoGame
    raw_game_data = data_loader.load_game_data(game) 
    
    print(f"Preparing dataset for {game_name}. Loaded raw data structure with {len(raw_game_data)} item(s).")
    
    # Convert to training examples and build lookup map
    examples = []
    prompt_to_ground_truth: Dict[str, List[str]] = {}
    
    if game_name == "cluedo":
        if len(raw_game_data) > 0 and "board_states" in raw_game_data[0]:
            all_interactions = raw_game_data[0]["board_states"]
            
            valid_states_count = 0
            for state in all_interactions:
                # Only use memory_update with non-null logged_reward for training
                if (state.get("interaction_type") == "memory_update" and 
                    state.get("logged_reward") is not None):
                    valid_states_count += 1
                    prompt = game.get_state_representation(state)
                    
                    # Original chosen response is the completion for GRPO KL term
                    chosen_response = state.get("chosen_response", {})
                    if isinstance(chosen_response, dict) and "summary" in chosen_response:
                        completion = chosen_response.get("summary", "") 
                    elif isinstance(chosen_response, dict) and "deducedCards" in chosen_response:
                        summary = chosen_response.get("summary", "No summary provided")
                        deduced_cards = chosen_response.get("deducedCards", [])
                        completion = f"{summary} Deduced cards: {', '.join(deduced_cards)}"
                    else:
                        completion = json.dumps(chosen_response)

                    reward = float(state.get("logged_reward", 0.0))
                    ground_truth = state.get("ground_truth_deductions") 
                    # Ensure ground_truth is a list of strings, even if null/empty
                    if ground_truth is None:
                        ground_truth = []
                    elif not isinstance(ground_truth, list):
                        ground_truth = [] # Or handle error
                        
                    examples.append({
                        "prompt": prompt,
                        "completion": completion,
                        "reward": reward
                    })
                    # Add to lookup map, handling potential duplicate prompts if necessary
                    if prompt not in prompt_to_ground_truth: 
                         prompt_to_ground_truth[prompt] = ground_truth
                    # else: Handle duplicate prompts? For now, overwrite/ignore.
            
            print(f"Found {valid_states_count} valid memory_update interactions with rewards for training.")
            print(f"Created {len(examples)} training examples and {len(prompt_to_ground_truth)} prompt-to-ground-truth mappings.")
            
        else:
            print(f"Warning: Cluedo game data not in expected format: {raw_game_data}")
    else:
        # Standard processing for other games
        for game_data in raw_game_data:
            board_states = game_data.get("board_states", [])
            
            # For each state, create a training example
            for i in range(len(board_states) - 1):
                current_state = board_states[i]
                next_state = board_states[i + 1]
                
                # Skip if it's not the agent's turn
                if current_state.get("player") != "model":
                    continue
                
                # Create prompt for the current state
                prompt = f"Game: {game_name}\nCurrent board:\n"
                prompt += game.get_state_representation(current_state)
                prompt += "\nChoose your next move from the valid actions:"
                
                valid_actions = game.get_valid_actions(current_state)
                prompt += ", ".join(valid_actions)
                prompt += "\nYour move: "
                
                # Get the action that was actually taken
                next_position = next_state.get("move", {}).get("position", [0, 0])
                completion = f"{next_position[0]},{next_position[1]}"
                
                examples.append({
                    "prompt": prompt,
                    "completion": completion
                })
        
        print(f"Created {len(examples)} training examples (non-Cluedo game).")

    # Convert examples list to Hugging Face dataset dict
    dataset_dict = {
        "prompt": [ex["prompt"] for ex in examples],
        "completion": [ex["completion"] for ex in examples],
        "reward": [ex.get("reward", 0.0) for ex in examples] # Include original rewards
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset, prompt_to_ground_truth

def create_reward_function(
    game_name: str,
    reward_name: str,
    reward_config: Dict[str, Any],
    prompt_to_ground_truth: Dict[str, List[str]]
) -> Callable:
    """
    Create a reward function for GRPO training.
    For Cluedo, it compares generated deductions against ground truth.
    
    Args:
        game_name: Name of the game
        reward_name: Name of the reward model
        reward_config: Configuration for the reward model
        prompt_to_ground_truth: Lookup map from prompt to ground truth deductions
        
    Returns:
        Reward function that takes completions and returns rewards
    """
    # Initialize game and reward model (optional, might not be needed anymore)
    # game = GameRegistry.get_game(game_name, {})
    # reward_model = RewardRegistry.get_reward_model(reward_name, reward_config)
    
    def reward_function(
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.
        For Cluedo, compares model's deduced cards against ground truth.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings (model's generated JSON response)
            
        Returns:
            List of reward values
        """
        batch_rewards = []
        
        for prompt, completion_str in zip(prompts, completions):
            reward = 0.0 # Default reward
            
            # Only calculate custom reward for Cluedo memory updates
            # Check if the prompt is one we have ground truth for
            if game_name == "cluedo" and prompt in prompt_to_ground_truth:
                ground_truth_deductions = set(prompt_to_ground_truth[prompt])
                model_deductions = set()
                
                try:
                    # Parse the model's generated JSON completion
                    parsed_completion = json.loads(completion_str)
                    
                    # Extract deduced cards - check common key names
                    if isinstance(parsed_completion, dict):
                        if "deducedCards" in parsed_completion and isinstance(parsed_completion["deducedCards"], list):
                            model_deductions = set(parsed_completion["deducedCards"])
                        elif "newly_deduced_held_cards" in parsed_completion and isinstance(parsed_completion["newly_deduced_held_cards"], list):
                             model_deductions = set(parsed_completion["newly_deduced_held_cards"])
                             
                    # Compare sets for exact match
                    if model_deductions == ground_truth_deductions:
                        # Reward more if the deduction was non-trivial (not empty)
                        reward = 1.0 if ground_truth_deductions else 0.1 
                    else:
                        # Optional: Implement partial reward based on overlap (e.g., Jaccard index)
                        # intersection = len(model_deductions.intersection(ground_truth_deductions))
                        # union = len(model_deductions.union(ground_truth_deductions))
                        # reward = intersection / union if union > 0 else 0.0
                        reward = 0.0 # Penalize incorrect deductions for now
                        
                except json.JSONDecodeError:
                    # Penalize completions that aren't valid JSON
                    reward = -0.1 
                except Exception as e:
                    print(f"Error during reward calculation for prompt: {e}")
                    reward = 0.0 # Assign neutral reward on unexpected error
            
            # else: Handle non-cluedo games or prompts not in lookup? 
            # For now, assign 0.0 reward if not a Cluedo prompt we have ground truth for.
            
            batch_rewards.append(reward)
            
        # print(f"Calculated batch rewards: {batch_rewards[:5]}...")
        return batch_rewards
    
    return reward_function

def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Ensure model parameters are trainable
    model.train() # Set model to training mode
    for param in model.parameters():
        param.requires_grad = True
    print("Set model parameters to requires_grad=True")
    
    # Prepare dataset and prompt-to-ground-truth lookup
    dataset, prompt_to_ground_truth = prepare_game_dataset(args.game, args.data_dir, tokenizer)
    
    # Create reward function, passing the lookup map
    reward_config = { # Config might be needed by reward_model if used in fallback
        # Add any relevant config if needed
    }
    reward_func = create_reward_function(
        args.game, 
        args.reward, 
        reward_config, 
        prompt_to_ground_truth # Pass the map here
    )
    
    # Configure GRPO training
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        fp16=False,  # Keep fp16 disabled for now, enable later if GPU supports it
        optim="adamw_torch",
        remove_unused_columns=False,
        num_generations=2,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_func,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
