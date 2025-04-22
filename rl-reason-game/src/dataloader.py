import json
import os
from typing import Dict, Any, List, Optional
import glob
from abstract.game import BaseGame

class DataLoader:
    """
    Utility class for loading game data from JSON files.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing game data JSON files
        """
        self.data_dir = data_dir
        
    def load_game_data(self, game: BaseGame, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """
        Load and parse game data for a specific game.
        
        Args:
            game: Game instance to use for parsing
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            List of parsed game data dictionaries
        """
        # Check if the game instance has a custom loading method
        if hasattr(game, 'custom_load_game_data') and callable(getattr(game, 'custom_load_game_data')):
            print(f"Using custom data loading method for {game.__class__.__name__}")
            return game.custom_load_game_data(self.data_dir)
        
        # Continue with default loading if no custom method
        file_pattern = os.path.join(self.data_dir, pattern)
        json_files = glob.glob(file_pattern)
        
        if not json_files:
            raise ValueError(f"No JSON files found matching pattern '{file_pattern}'")
        
        parsed_data = []
        for file_path in json_files:
            try:
                # Load JSON file
                with open(file_path, 'r') as f:
                    game_data = json.load(f)
                
                # Parse using the game's parser
                parsed_game_data = game.parse_game_result(game_data)
                parsed_data.append(parsed_game_data)
                
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
        
        return parsed_data
    

    def save_game_data(self, data: Dict[str, Any], filename: str) -> None:
        """
        Save game data to a JSON file.
        
        Args:
            data: Game data to save
            filename: Target filename
        """
        file_path = os.path.join(self.data_dir, filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
