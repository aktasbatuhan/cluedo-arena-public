import { Game } from './models/Game.js';
import { HumanAgent } from './models/HumanAgent.js';

export async function runGame(game) {
  if (!game) {
    throw new Error('Game instance is required');
  }

  while (!game.winner && game.activePlayers > 1) {
    await game.processTurn();
    
    // If it's human player's turn, wait for their action
    if (game.agents[game.activeAgentIndex] instanceof HumanAgent) {
      console.log('Waiting for human player action...');
      return; // Exit and wait for human action
    }
    
    // Add a small delay between turns
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
} 