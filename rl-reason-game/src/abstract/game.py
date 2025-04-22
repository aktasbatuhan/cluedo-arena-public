from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseGame(ABC):
    """
    Abstract base class for all game implementations.
    """
    
    def __init__(self, metadata: Dict[str, Any]):
        """
        Initialize the game with configuration parameters.
        
        Args:
            config: Dictionary containing game-specific configuration
        """
        self.metadata = metadata
        self.data = None
    
    @abstractmethod
    def parse_game_result(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the game result data from a JSON file and saves to class state.
        
        Args:
            game_data: Dictionary containing the game result data
            
        Returns:
            Processed game data in a standardized format
        """
        pass
    
    @abstractmethod
    def get_state_representation(self, game_state: Dict[str, Any]) -> str:
        """
        Convert a game state into a string representation suitable for LLM input.
        
        Args:
            game_state: Dictionary containing the game state
            
        Returns:
            String representation of the game state
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, game_state: Dict[str, Any]) -> List[str]:
        """
        Get the list of valid actions for a given game state.
        
        Args:
            game_state: Dictionary containing the game state
            
        Returns:
            List of valid action strings
        """
        pass
    
    @abstractmethod
    def is_terminal_state(self, game_state: Dict[str, Any]) -> bool:
        """
        Check if the given game state is terminal (game over).
        
        Args:
            game_state: Dictionary containing the game state
            
        Returns:
            True if the state is terminal, False otherwise
        """
        pass
