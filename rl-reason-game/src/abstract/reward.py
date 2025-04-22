from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

class BaseRewardModel(ABC):
    """
    Abstract base class for reward models.
    
    Reward models convert game outcomes and trajectories into reward signals
    for training reinforcement learning agents.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the reward model with configuration parameters.
        
        Args:
            config: Dictionary containing reward-specific configuration
        """
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def compute_reward(self, game_data: Dict[str, Any], agent_id: str) -> float:
        """
        Compute the overall reward for an agent based on a complete game.
        
        Args:
            game_data: Dictionary containing the game data
            agent_id: Identifier of the agent (e.g., "X" or "O")
            
        Returns:
            Scalar reward value
        """
        pass
    
    @abstractmethod
    def compute_step_rewards(self, game_data: Dict[str, Any], agent_id: str) -> List[float]:
        """
        Compute rewards for each step in a game trajectory.
        
        Args:
            game_data: Dictionary containing the game data
            agent_id: Identifier of the agent (e.g., "X" or "O")
            
        Returns:
            List of reward values, one for each step in the game
        """
        pass
    
    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns from rewards.
        
        Args:
            rewards: List of rewards for each step
            gamma: Discount factor
            
        Returns:
            List of discounted returns
        """
        returns = []
        G = 0
        
        # Compute returns in reverse order
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
            
        return returns
    
    def normalize_rewards(self, rewards: List[float], epsilon: float = 1e-8) -> List[float]:
        """
        Normalize rewards to have zero mean and unit variance.
        
        Args:
            rewards: List of rewards
            epsilon: Small constant to avoid division by zero
            
        Returns:
            Normalized rewards
        """
        import numpy as np
        rewards_array = np.array(rewards)
        
        if len(rewards) <= 1:
            return rewards
        
        mean = np.mean(rewards_array)
        std = np.std(rewards_array)
        
        if std < epsilon:
            return [0.0] * len(rewards)
            
        normalized_rewards = (rewards_array - mean) / (std + epsilon)
        
        return normalized_rewards.tolist()
