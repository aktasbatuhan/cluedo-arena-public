from abstract.game import BaseGame
from game_registry import GameRegistry
from typing import Any, Dict, List, Optional
import json
import os
import glob

# TODO: Define the structure of the Cluedo game state if necessary
# type CluedoGameState = Dict[str, Any] 

@GameRegistry.register("cluedo")
class CluedoGame(BaseGame):
    """
    Implementation of the Cluedo game for the RL-Reason-Game framework.
    Handles parsing game interaction data, representing states, and determining valid actions.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Standard Cluedo items
        self.suspects = [
            'Miss Scarlet',
            'Colonel Mustard',
            'Mrs. White',
            'Mr. Green',
            'Mrs. Peacock',
            'Professor Plum'
        ]
        self.weapons = ['Candlestick', 'Dagger', 'Lead Pipe', 'Revolver', 'Rope', 'Wrench']
        self.rooms = ['Kitchen', 'Ballroom', 'Conservatory', 'Dining Room', 'Billiard Room', 'Library', 'Lounge', 'Hall', 'Study']
        self.all_cards = self.suspects + self.weapons + self.rooms
        print(f"Initialized CluedoGame with {len(self.suspects)} suspects, {len(self.weapons)} weapons, {len(self.rooms)} rooms.")

    def load_jsonl_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Custom method to load JSONL data for Cluedo interactions.
        
        Args:
            data_dir: Directory containing the cluedo_interactions.jsonl file
            
        Returns:
            List of parsed game data dictionaries
        """
        file_pattern = os.path.join(data_dir, "cluedo_interactions.jsonl")
        print(f"Looking for Cluedo data at: {file_pattern}")
        
        if not os.path.exists(file_pattern):
            # List all files in the directory to help with debugging
            try:
                files_in_dir = os.listdir(data_dir)
                print(f"Files found in {data_dir}: {files_in_dir}")
            except Exception as e:
                print(f"Error listing directory: {str(e)}")
            
            raise ValueError(f"Cluedo data file not found at: {file_pattern}")
        
        line_count = 0
        memory_update_count = 0
        valid_reward_count = 0
        parsed_data = []
        
        with open(file_pattern, 'r') as f:
            for line in f:
                line_count += 1
                try:
                    game_data = json.loads(line.strip())
                    # Count memory_update interactions
                    if game_data.get('interaction_type') == 'memory_update':
                        memory_update_count += 1
                        # Count valid rewards
                        if game_data.get('logged_reward') is not None:
                            valid_reward_count += 1
                    
                    parsed_game_data = self.parse_game_result(game_data)
                    if parsed_game_data:  # Only add if valid (non-empty dict)
                        parsed_data.append(parsed_game_data)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line {line_count}: {str(e)}")
                except Exception as e:
                    print(f"Error processing line {line_count}: {str(e)}")
        
        print(f"Loaded {len(parsed_data)} valid game interactions from JSONL")
        print(f"Stats: Total lines: {line_count}, memory_updates: {memory_update_count}, with valid rewards: {valid_reward_count}")
        
        # Convert to the format expected by the trainer
        # The trainer expects a list of dictionaries with a "board_states" key
        # We'll convert our interaction data to match this format
        converted_data = [{
            "board_states": parsed_data
        }]
        
        return converted_data
    
    # Custom method to override default DataLoader.load_game_data behavior
    # This will be called by DataLoader when trying to load game data
    def custom_load_game_data(self, data_dir: str) -> List[Dict[str, Any]]:
        """
        Custom method that overrides the default game data loading for Cluedo.
        
        Args:
            data_dir: Directory containing the cluedo_interactions.jsonl file
            
        Returns:
            List of parsed game data dictionaries
        """
        return self.load_jsonl_data(data_dir)

    def parse_game_result(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the raw game interaction data from cluedo_interactions.jsonl.
        The data is already structured, so we mostly return it as is.
        
        Args:
            game_data: A dictionary representing a single interaction/turn from the JSONL file.

        Returns:
            The input dictionary, potentially validated or slightly restructured if needed.
        """
        # Basic validation (optional)
        required_keys = ['interaction_type', 'prompt', 'chosen_response']
        if not all(key in game_data for key in required_keys):
            print(f"Warning: Game data missing required keys: {required_keys}. Got: {list(game_data.keys())}")
            # Handle error or return a default structure
            return {}

        # For now, return the dictionary directly as the framework might use its fields
        # print(f"Parsed game data for interaction type: {game_data.get('interaction_type')}")
        return game_data

    def get_state_representation(self, game_state: Dict[str, Any]) -> str:
        """
        Extracts the LLM prompt from the parsed game state dictionary.
        
        Args:
            game_state: The dictionary returned by parse_game_result.

        Returns:
            A string representation of the game state (the LLM prompt).
        """
        # The game_state here is the dictionary parsed from the JSONL line
        prompt = game_state.get('prompt')
        if prompt is None:
            print("Warning: 'prompt' key not found in game_state dictionary.")
            return ""
        if not isinstance(prompt, str):
            print(f"Warning: 'prompt' value is not a string (type: {type(prompt)}).")
            return str(prompt) # Attempt to cast
            
        # print(f"Returning prompt for state (type: {game_state.get('interaction_type')})")
        return prompt

    def get_valid_actions(self, game_state: Any) -> List[str]:
        """
        Determines the set of valid actions (e.g., suggestions, accusations) possible 
        from the given game state.
        
        NOTE: For GRPO training based on pre-recorded interactions, this might not be
        strictly needed during training if we use the recorded 'chosen_response' as the action.
        It could be useful for validation or post-training evaluation/play.

        Args:
            game_state: The current game state.

        Returns:
            A list of strings representing valid actions (currently placeholder).
        """
        # TODO: Implement actual logic if needed for evaluation or active play.
        # For now, returning empty as the action comes from 'chosen_response' in the data.
        # print(f"Getting valid actions (placeholder) for state: {type(game_state)}")
        return [] # Placeholder

    def is_terminal_state(self, game_state: Any) -> bool:
        """
        Checks if the given game interaction represents the end of the game.
        
        NOTE: Based on the data format, this might correspond to a specific 
        'interaction_type' like 'accusation_result' or similar, or checking the content
        of 'consider_accusation' or 'evaluate_challenge'. This needs clarification
        based on the full dataset or game logic.

        Args:
            game_state: The current game state/interaction data.

        Returns:
            True if the state is terminal, False otherwise (currently placeholder).
        """
        # TODO: Implement actual logic based on how game end is represented in the data.
        # Example: Check if interaction_type indicates a final, correct accusation.
        # print(f"Checking terminal state (placeholder) for state: {type(game_state)}")
        interaction_type = game_state.get('interaction_type')
        # Simplistic check: did the agent decide to make a final accusation?
        if interaction_type == 'consider_accusation':
             response = game_state.get('chosen_response', {})
             if response.get('shouldAccuse') is True:
                 # We might need more info from the dataset to know if it was *correct*
                 # For now, just treating the decision to accuse as potentially terminal
                 # print("Terminal state detected: Agent decided to accuse.")
                 return True 
        return False # Placeholder

    # Add any other Cluedo-specific helper methods needed 