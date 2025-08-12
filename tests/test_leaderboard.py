import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT / 'cluedo_game_engine')
sys.path.append(str(ROOT / 'scripts' / 'utility'))

from leaderboard import analyze_games


def test_analyze_games_basic():
    data = [{
        'winner': {'model': 'ModelA'},
        'totalTurns': 8,
        'players': [
            {'model': 'ModelA', 'name': 'A', 'missedOpportunities': [{'turn': 2}]},
            {'model': 'ModelB', 'name': 'B', 'missedOpportunities': []}
        ]
    }]

    stats = analyze_games(data)
    assert stats['ModelA']['games_played'] == 1
    assert stats['ModelA']['games_won'] == 1
    assert stats['ModelA']['avg_completion_time'] == 8
    assert stats['ModelA']['risk_aversion_score'] == 6
    assert stats['ModelB']['games_played'] == 1
    assert stats['ModelB']['games_won'] == 0

