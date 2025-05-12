import { Game } from './models/Game.js';
import { startServer } from './server.js';
import { LLMService } from './services/llm.js';

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
  // Check command line arguments
  const runGamesArg = process.argv.find(arg => arg === '--run-games');
  const numGamesArg = process.argv.find(arg => arg.startsWith('--num-games='));
  const backendArg = process.argv.find(arg => arg.startsWith('--llm-backend='));

  // Set LLM Backend based on argument
  if (backendArg) {
      const backendName = backendArg.split('=')[1];
      if (backendName === 'openrouter' || backendName === 'cohere') {
          try {
              LLMService.setBackend(backendName);
          } catch (error) {
               console.error(`Error setting LLM backend: ${error.message}`);
               process.exit(1);
          }
      } else {
          console.error(`Invalid --llm-backend specified: ${backendName}. Use 'cohere' or 'openrouter'.`);
          process.exit(1);
      }
  } else {
      // Default to cohere if no argument provided
      // Ensure CO_API_KEY is set for default
      try {
        LLMService.setBackend('cohere'); 
      } catch (error) {
        // Don't exit if default fails, maybe server mode doesn't need LLM immediately
        // But log a warning if keys might be needed later
        console.warn(`Warning: Defaulting to Cohere backend, but setup failed: ${error.message}. Ensure keys are set if running games.`);
      }
  }

  if (runGamesArg) {
    // Extract number of games from arguments or default to 1
    const numGames = numGamesArg 
      ? parseInt(numGamesArg.split('=')[1], 10) 
      : 1;
      
    await runAutomatedGames(numGames);
  } else {
    // Default behavior - start server
    try {
      await startServer();
      console.log('Server running - Open http://localhost:3000 to play');
    } catch (error) {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
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