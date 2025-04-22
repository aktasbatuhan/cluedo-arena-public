from abstract.reward import BaseRewardModel
from reward_registry import RewardRegistry
from typing import Any, Dict, List, Optional
import numpy as np

@RewardRegistry.register("cluedo_deduction")
class CluedoRewardModel(BaseRewardModel):
    """
    Reward model specifically for the Cluedo game interactions.
    Calculates rewards based on the quality of deductions for 'memory_update' interactions.
    Uses the pre-calculated 'logged_reward' from the dataset only for these interactions.
    Returns None for interactions that should not be used for training.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        print("Initialized CluedoRewardModel (Training only on valid memory_updates)")
        self.reward_key = 'logged_reward' # Key in the dataset containing the reward
        self.target_interaction_type = 'memory_update'

    def compute_reward(self, game_data: Dict[str, Any], agent_id: Optional[Any] = None) -> Optional[float]:
        """
        Computes a reward only for valid 'memory_update' interactions.
        
        Args:
            game_data: The dictionary representing the interaction data.
            agent_id: Identifier for the agent (optional).

        Returns:
            A float reward if the interaction is a valid 'memory_update', otherwise None.
        """
        interaction_type = game_data.get('interaction_type')

        # Only process the target interaction type
        if interaction_type != self.target_interaction_type:
            # print(f"Skipping reward calculation: Interaction type is '{interaction_type}', not '{self.target_interaction_type}'.")
            return None 

        # Check for the presence and validity of the logged reward
        reward = game_data.get(self.reward_key)
        
        if reward is not None:
            try:
                # Ensure reward is a float
                return float(reward)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert logged_reward '{reward}' for {self.target_interaction_type}. Returning None.")
                return None
        else:
            # Logged reward is null for the target interaction type - skip this data point
            # print(f"Skipping reward calculation: '{self.reward_key}' is null for interaction type '{interaction_type}'.")
            return None

    def compute_step_rewards(self, game_data: List[Dict[str, Any]], agent_id: Optional[Any] = None) -> np.ndarray:
        """
        Computes rewards for a sequence of game interactions, returning np.nan for skipped steps.

        Args:
            game_data: A list of dictionaries, each representing an interaction step.
            agent_id: Identifier for the agent (optional).

        Returns:
            A numpy array of rewards (float or np.nan), one for each step in the sequence.
        """
        rewards = []
        if isinstance(game_data, list):
            for step_data in game_data:
                reward = self.compute_reward(step_data, agent_id)
                rewards.append(reward if reward is not None else np.nan)
        elif isinstance(game_data, dict):
            reward = self.compute_reward(game_data, agent_id)
            rewards.append(reward if reward is not None else np.nan)
        else:
            print(f"Warning: Unexpected type for game_data in compute_step_rewards: {type(game_data)}")
            
        return np.array(rewards, dtype=np.float32) # np.nan is float

    # Override compute_returns or normalize_rewards if custom logic is needed,
    # otherwise the base class implementations will be used. 