import json
from collections import defaultdict
from typing import Dict, List

def analyze_games(data: List[dict]) -> Dict[str, dict]:
    stats = defaultdict(lambda: {
        'games_played': 0,
        'games_won': 0,
        'total_turns': [],
        'missed_opportunities': [],
        'risk_aversion_sum': 0
    })
    
    # Process each game
    for game in data:
        # Skip games with no winner (incomplete games)
        winner = game.get('winner', {})
        if not isinstance(winner, dict) or not winner.get('model'):
            continue
            
        total_turns = game.get('totalTurns', 0)
        winner_model = winner.get('model', '')
        
        # Track player-specific turn counts and model data
        for player in game.get('players', []):
            if player.get('model'):
                model = player['model']
                player_name = player.get('name', '')
                
                # Count this game for the model
                stats[model]['games_played'] += 1
                
                # Mark game as won if this model was the winner
                if winner_model and model == winner_model:
                    stats[model]['games_won'] += 1
                    # Only record turn counts for games this model won
                    stats[model]['total_turns'].append(total_turns)
                
                # Process missed opportunities for this player
                for missed in player.get('missedOpportunities', []):
                    turn_diff = total_turns - missed.get('turn', 0)
                    stats[model]['risk_aversion_sum'] += turn_diff
                    stats[model]['missed_opportunities'].append(turn_diff)

    # Calculate final statistics
    results = {}
    for model, data in stats.items():
        turns = data['total_turns']
        results[model] = {
            'model_name': model,
            'games_played': data['games_played'],
            'games_won': data['games_won'],
            'win_rate': round(data['games_won'] / data['games_played'] * 100, 2) if data['games_played'] > 0 else 0,
            'avg_completion_time': round(sum(turns) / len(turns), 2) if turns else 0,
            'longest_game': max(turns) if turns else 0,
            'shortest_game': min(turns) if turns else 0,
            'risk_aversion_score': round(data['risk_aversion_sum'] / data['games_played'], 2) if data['games_played'] > 0 else 0,
            'missed_opportunities_count': len(data['missed_opportunities'])
        }
    
    return results

def print_leaderboard(stats: Dict[str, dict]):
    # Convert to list and sort by win rate
    leaderboard = list(stats.values())
    leaderboard.sort(key=lambda x: (-x['win_rate'], x['avg_completion_time']))
    
    # Print header
    print("\n=== CLUE GAME LEADERBOARD ===")
    print(f"{'Model':<25} {'Games':<8} {'Wins':<8} {'Win %':<8} {'Avg Turns':<10} {'Shortest':<10} {'Longest':<10} {'Risk Score':<12}")
    print("=" * 90)
    
    # Print each row
    for entry in leaderboard:
        print(
            f"{entry['model_name']:<25} "
            f"{entry['games_played']:<8} "
            f"{entry['games_won']:<8} "
            f"{entry['win_rate']}%{' ':<4} "
            f"{entry['avg_completion_time']:<10} "
            f"{entry['shortest_game']:<10} "
            f"{entry['longest_game']:<10} "
            f"{entry['risk_aversion_score']:<12}"
        )

# Load and process the data
with open('game_results.json', 'r') as file:
    game_data = json.load(file)

# Generate and display the leaderboard
stats = analyze_games(game_data)
print_leaderboard(stats)