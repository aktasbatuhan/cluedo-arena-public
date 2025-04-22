from typing import Dict, Type, Any
from abstract.reward import BaseRewardModel
from abstract.registry import BaseRegistry

class RewardRegistry(BaseRegistry[BaseRewardModel]):
    """
    Registry for reward model implementations.
    
    This class allows for registering and retrieving reward model implementations
    in a centralized manner.
    """
    
    _registry: Dict[str, Type[BaseRewardModel]] = {}
    
    @classmethod
    def get_reward_model(cls, reward_name: str, config: Dict[str, Any]) -> BaseRewardModel:
        """
        Get an instance of a registered reward model.
        
        Args:
            reward_name: Name of the registered reward model
            config: Configuration for the reward model instance
            
        Returns:
            Instance of the requested reward model
            
        Raises:
            ValueError: If the reward model name is not registered
        """
        return cls.get_implementation(reward_name, config)
    
    @classmethod
    def list_reward_models(cls) -> list:
        """
        List all registered reward models.
        
        Returns:
            List of registered reward model names
        """
        return cls.list_implementations()
