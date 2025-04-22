from typing import Dict, Type, Any
from abstract.game import BaseGame
from abstract.registry import BaseRegistry

class GameRegistry(BaseRegistry[BaseGame]):
    """
    Registry for game implementations.
    
    This class allows for registering and retrieving game implementations
    in a centralized manner.
    """
    
    _registry: Dict[str, Type[BaseGame]] = {}
    
    @classmethod
    def get_game(cls, game_name: str, config: Dict[str, Any]) -> BaseGame:
        """
        Get an instance of a registered game.
        
        Args:
            game_name: Name of the registered game
            config: Configuration for the game instance
            
        Returns:
            Instance of the requested game
            
        Raises:
            ValueError: If the game name is not registered
        """
        return cls.get_implementation(game_name, config)
    
    @classmethod
    def list_games(cls) -> list:
        """
        List all registered games.
        
        Returns:
            List of registered game names
        """
        return cls.list_implementations()
