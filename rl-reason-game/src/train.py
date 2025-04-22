#!/usr/bin/env python3

import os
import argparse
from typing import Dict, Any, List, Optional, Callable
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
) -> Dataset:
    """
    Prepare a dataset from game data for training.
    
    Args:
        game_name: Name of the game
        data_dir: Directory containing game data
        tokenizer: Tokenizer for the model
        
    Returns:
        Dataset formatted for GRPO training
    """
    # Initialize game with empty config
    game = GameRegistry.get_game(game_name, {})
    
    # Load game data
    data_loader = DataLoader(data_dir)
    game_data_list = data_loader.load_game_data(game)
    
    print(f"Preparing dataset for {game_name}. Loaded {len(game_data_list)} items.")
    
    # Convert to training examples
    examples = []
    
    # If this is the Cluedo game, handle its specific data format
    if game_name == "cluedo":
        # We expect a list with one entry containing a key "board_states" with all interactions
        if len(game_data_list) > 0 and "board_states" in game_data_list[0]:
            board_states = game_data_list[0]["board_states"]
            
            # Filter to only include memory_update interactions with valid rewards
            valid_states = []
            for state in board_states:
                # Only use memory_update with non-null logged_reward for training
                if (state.get("interaction_type") == "memory_update" and 
                    state.get("logged_reward") is not None):
                    valid_states.append(state)
            
            print(f"Filtered to {len(valid_states)} valid memory_update interactions with rewards")
            
            for state in valid_states:
                prompt = game.get_state_representation(state)
                # Get the expected completion (response)
                chosen_response = state.get("chosen_response", {})
                
                # Try to extract summary from chosen_response
                if isinstance(chosen_response, dict) and "summary" in chosen_response:
                    completion = chosen_response.get("summary", "")
                # If no summary directly, try to get it from deducedCards structure if present
                elif isinstance(chosen_response, dict) and "deducedCards" in chosen_response:
                    # Format the completion to be consistent
                    summary = chosen_response.get("summary", "No summary provided")
                    deduced_cards = chosen_response.get("deducedCards", [])
                    completion = f"{summary} Deduced cards: {', '.join(deduced_cards)}"
                else:
                    # Fallback: convert the whole chosen_response to JSON
                    completion = json.dumps(chosen_response)
                
                examples.append({
                    "prompt": prompt,
                    "completion": completion,
                    "reward": float(state.get("logged_reward", 0.0))
                })
        else:
            print(f"Warning: Cluedo game data not in expected format: {game_data_list}")
    else:
        # Standard processing for other games
        for game_data in game_data_list:
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
    
    print(f"Created {len(examples)} training examples")
    
    # Convert to Hugging Face dataset
    dataset_dict = {
        "prompt": [ex["prompt"] for ex in examples],
        "completion": [ex["completion"] for ex in examples]
    }
    
    # Add rewards if available
    if examples and "reward" in examples[0]:
        dataset_dict["reward"] = [ex.get("reward", 0.0) for ex in examples]
    
    return Dataset.from_dict(dataset_dict)

def create_reward_function(
    game_name: str,
    reward_name: str,
    reward_config: Dict[str, Any]
) -> Callable:
    """
    Create a reward function for GRPO training based on the specified reward model.
    
    Args:
        game_name: Name of the game
        reward_name: Name of the reward model
        reward_config: Configuration for the reward model
        
    Returns:
        Reward function that takes completions and returns rewards
    """
    # Initialize game and reward model
    game = GameRegistry.get_game(game_name, {})
    reward_model = RewardRegistry.get_reward_model(reward_name, reward_config)
    
    def reward_function(
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.
        Uses pre-computed rewards from the dataset if available.
        
        Args:
            prompts: List of prompt strings
            completions: List of completion strings (agent responses)
            rewards: List of pre-computed rewards from the dataset (passed via kwargs)
            
        Returns:
            List of reward values
        """
        
        # Check if the rewards are passed directly in kwargs (this happens with GRPO when
        # the dataset contains reward values)
        if "rewards" in kwargs and isinstance(kwargs["rewards"], (list, np.ndarray)) and len(kwargs["rewards"]) == len(prompts):
            # print(f"Using pre-computed rewards from dataset: {kwargs['rewards'][:5]}...")
            # Ensure rewards are floats
            try:
                return [float(r) for r in kwargs["rewards"]]
            except (ValueError, TypeError) as e:
                 print(f"Warning: Error converting pre-computed rewards to float: {e}. Returning zeros.")
                 return [0.0] * len(prompts)
        
        # Fallback if pre-computed rewards are not available or invalid
        print("Warning: Pre-computed rewards not found or invalid in kwargs. Using fallback reward calculation.")
        
        rewards = []
        # For Cluedo, if we somehow don't have pre-computed rewards, we might evaluate the completion
        if game_name == "cluedo":
            # Placeholder: Evaluate completion quality (e.g., is it valid JSON?)
            # This part needs a more robust implementation if used.
            for completion in completions:
                try:
                    # Example: Reward valid JSON summaries
                    parsed_completion = json.loads(completion)
                    if isinstance(parsed_completion, dict) and "summary" in parsed_completion:
                         rewards.append(0.1) # Small reward for valid format
                    else:
                         rewards.append(0.0)
                except json.JSONDecodeError:
                    rewards.append(-0.1) # Penalize invalid JSON
            print(f"Using fallback Cluedo reward calculation, generated rewards: {rewards[:5]}...")
            return rewards
            
        # Original board game reward calculation (likely won't be reached for Cluedo)
        for prompt, completion in zip(prompts, completions):
            try:
                lines = prompt.split("\n")
                current_board_idx = lines.index("Current board:") + 1
                board_rep = "\n".join(lines[current_board_idx:current_board_idx+5])
                board = [[" " for _ in range(3)] for _ in range(3)]
                current_state = {"board": board}
                
                try:
                    row, col = map(int, completion.strip().split(","))
                    next_board = [r.copy() for r in board]
                    next_board[row][col] = "X"
                    next_state = {"board": next_board}
                    is_terminal = game.is_terminal_state(next_state)
                    
                    if is_terminal:
                        winner = game._check_winner(next_board) # Assuming _check_winner exists
                        game_result = {"result": winner if winner else "draw", "players": ["X", "O"]}
                        reward = reward_model.compute_reward(game_result, "X")
                    else:
                        reward = 0.1
                except (ValueError, IndexError):
                    reward = -1.0
            except Exception as e:
                print(f"Error computing fallback reward: {str(e)}")
                reward = 0.0
            rewards.append(reward)
        
        return rewards
    
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
    
    # Prepare dataset
    dataset = prepare_game_dataset(args.game, args.data_dir, tokenizer)
    
    # Create reward function
    reward_config = {
        "win_reward": 1.0,
        "loss_reward": -1.0,
        "draw_reward": 0.2,
        "step_penalty": -0.05
    }
    reward_func = create_reward_function(args.game, args.reward, reward_config)
    
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
        gradient_checkpointing=True,
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
