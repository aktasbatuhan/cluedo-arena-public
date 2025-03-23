import { Game } from './models/Game.js';
import { startServer } from './server.js';

async function runAutomatedGames(numGames) {
  console.log(`Starting ${numGames} automated games...`);
  
  for (let i = 0; i < numGames; i++) {
    console.log(`\n=== Starting Game ${i + 1}/${numGames} ===\n`);
    const game = new Game('spectate');
    await game.initialize();
    
    while (!game.isGameOver()) {
      await game.processTurn();
    }
    
    console.log(`Game ${i + 1} completed`);
  }
}

async function main() {
  try {
    await startServer();
    console.log('Server running - Open http://localhost:3000 to play');
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Only run main if not in test mode
if (process.env.NODE_ENV !== 'test') {
  main().catch(error => {
    console.error('Unhandled error:', error);
    process.exit(1);
  });
}

process.on('unhandledRejection', (error) => {
  console.error('Unhandled promise rejection:', error);
  process.exit(1);
}); 