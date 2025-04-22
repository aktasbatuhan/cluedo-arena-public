# RL-Reason-Game

A framework for training reinforcement learning agents on reasoning games using Group Relative Policy Optimization (GRPO).

## Overview

RL-Reason-Game provides a flexible framework for training language models to play structured reasoning games (like Tic-tac-toe). The framework uses a registry-based architecture to support multiple game implementations and reward models, making it easy to extend with new games and training approaches.

Key features:
- Registry system for games and reward models
- Abstract base classes for extensibility
- GRPO-based training pipeline
- Data loading and processing utilities

## Project Structure

```
rl-reason-game/
├── src/
│   ├── abstract/
│   │   ├── game.py        # Base game interface
│   │   ├── registry.py    # Generic registry implementation
│   │   ├── reward.py      # Base reward model interface
│   ├── dataloader.py      # Utilities for loading game data
│   ├── game_registry.py   # Registry for game implementations
│   ├── reward_registry.py # Registry for reward model implementations
│   ├── train.py           # Main training script
├── data/
│   ├── games/             # Game data files
├── models/                # Trained model outputs
```

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/erdiari/rl-reason-game.git
cd rl-reason-game
```

This project uses `uv` as its package manager. If you don't have it installed, you can install it following the instructions at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

Then install the package and its dependencies:

```bash
uv pip install -e .
```

Alternatively, you can use pip:

```bash
pip install -e .
```

### Requirements

- Python 3.12 or higher (as specified in `.python-version`)
- Dependencies (managed by `uv`):
  - datasets>=3.5.0
  - numpy>=2.2.4
  - torch>=2.6.0
  - transformers>=4.51.3
  - trl>=0.16.1

## Usage

### Training an Agent

To train an agent using GRPO:

```bash
python src/train.py --game tictactoe --reward win_loss --model Qwen/Qwen2-0.5B-Instruct --output_dir models/tictactoe-agent
```

Command-line arguments:
- `--game`: Name of the registered game to train on (default: "tictactoe")
- `--reward`: Name of the reward model to use (default: "win_loss")
- `--data_dir`: Directory containing game data (default: "data/games")
- `--model`: Base LLM to fine-tune (default: "Qwen/Qwen2-0.5B-Instruct")
- `--output_dir`: Directory to save the trained model (default: "models/grpo-agent")
- `--learning_rate`: Learning rate for training (default: 1e-5)
- `--batch_size`: Batch size for training (default: 8)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)
- `--max_steps`: Maximum number of training steps (-1 for full epochs)

### Adding a New Game

To add a new game, create a new implementation that inherits from `BaseGame`:

```python
from abstract.game import BaseGame
from game_registry import GameRegistry

@GameRegistry.register("my_new_game")
class MyNewGame(BaseGame):
    def __init__(self, config):
        super().__init__(config)
        # Initialize game-specific attributes
        
    def parse_game_result(self, game_data):
        # Implement parsing logic
        pass
        
    def get_state_representation(self, game_state):
        # Implement state representation
        pass
        
    def get_valid_actions(self, game_state):
        # Implement valid action listing
        pass
        
    def is_terminal_state(self, game_state):
        # Implement terminal state checking
        pass
```

### Adding a New Reward Model

To add a new reward model, create a new implementation that inherits from `BaseRewardModel`:

```python
from abstract.reward import BaseRewardModel
from reward_registry import RewardRegistry

@RewardRegistry.register("my_reward")
class MyRewardModel(BaseRewardModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize reward-specific attributes
        
    def compute_reward(self, game_data, agent_id):
        # Implement overall reward calculation
        pass
        
    def compute_step_rewards(self, game_data, agent_id):
        # Implement step-by-step reward calculation
        pass
```

## Architecture

The framework is built around several key components:

### Registry System

The registry system provides a central location for registering and retrieving implementations:

- `BaseRegistry`: Generic registry implementation
- `GameRegistry`: Registry for game implementations
- `RewardRegistry`: Registry for reward model implementations

### Game Interface

The `BaseGame` abstract class defines the interface for all game implementations:

- `parse_game_result`: Parse game data from a JSON file
- `get_state_representation`: Convert a game state to a string representation
- `get_valid_actions`: Get the list of valid actions for a given state
- `is_terminal_state`: Check if a game state is terminal

### Reward Interface

The `BaseRewardModel` abstract class defines the interface for reward models:

- `compute_reward`: Calculate the overall reward for a game
- `compute_step_rewards`: Calculate rewards for each step in a game
- `compute_returns`: Calculate discounted returns from rewards
- `normalize_rewards`: Normalize rewards to have zero mean and unit variance

1. Load and prepare game data
2. Create a reward function based on the selected reward model
3. Configure and initialize the GRPO trainer
4. Train the model and save the result
