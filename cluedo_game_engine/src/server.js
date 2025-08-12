import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { Game } from './models/Game.js';
import { GameResult } from './models/GameResult.js';
import { spawnSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// In-memory store for completed games. Each element will be a comprehensive object.
const completedGames = [];
let game = null; // Single game instance for 'single' mode

/**
 * Computes leaderboard statistics by calling an external Python script.
 * @param {Array<Object>} results - An array of game result objects.
 * @returns {Object} An object containing the leaderboard stats.
 */
function computeLeaderboard(results) {
  const scriptPath = path.resolve(__dirname, '../../scripts/utility').replace(/\\/g, '/');
  const pyCode = `import json,sys,os\nsys.path.append('${scriptPath}')\nfrom leaderboard import analyze_games\ndata=json.loads(sys.stdin.read())\nprint(json.dumps(analyze_games(data)))`;

  const out = spawnSync('python', ['-c', pyCode], {
    input: JSON.stringify(results),
    encoding: 'utf-8'
  });

  if (out.status !== 0) {
    console.error('Leaderboard generation failed:', out.stderr);
    return {};
  }
  try {
    return JSON.parse(out.stdout);
  } catch (err) {
    console.error('Failed to parse leaderboard output:', err);
    return {};
  }
}

export async function startServer() {
  const app = express();
  const server = createServer(app);
  const io = new Server(server);

  app.use(express.static('public'));

  /**
   * Handles the result of a completed game.
   * - Stores the comprehensive result.
   * - Emits updates for game completion and leaderboard.
   * @param {Object} resultData - The data from the completed game.
   */
  function handleGameResult(resultData) {
    completedGames.push(resultData);

    // Emit the detailed log for this specific game
    io.emit('game-log', { ...resultData, gameIndex: completedGames.length });

    // Compute and emit the updated leaderboard
    const leaderboard = computeLeaderboard(completedGames);
    io.emit('leaderboard-update', leaderboard);
  }

  io.on('connection', (socket) => {
    // When a new client connects, send them the history
    socket.emit('past-games', completedGames);
    if (completedGames.length > 0) {
      socket.emit('leaderboard-update', computeLeaderboard(completedGames));
    }

    socket.on('game-mode', async (mode) => {
      try {
        if (mode === 'multi') {
          const numGames = 5; // Or make this configurable
          for (let i = 0; i < numGames; i++) {
            const multiGame = new Game('spectate', io); // UI will spectate
            await multiGame.initialize();
            const result = await runGameLoop(multiGame);
            handleGameResult(result);

            io.emit('game-progress', { total: numGames, completed: i + 1 });

            // Reset agent memories for the next game in the series
            if (multiGame.agents) {
              for (const agent of multiGame.agents) {
                await agent.memory.reset();
              }
            }
            await new Promise(resolve => setTimeout(resolve, 1000)); // Delay between games
          }
          return; // End after multi-game loop
        }

        // For 'single' mode
        game = new Game('single', io);
        await game.initialize();

        // Forward all necessary events from the game instance to the client
        game.on('suggestion', (data) => io.emit('game-event', { type: 'SUGGESTION', ...data }));
        game.on('challenge', (data) => io.emit('game-event', { type: 'CHALLENGE', ...data }));
        game.on('accusation', (data) => io.emit('game-event', { type: 'ACCUSATION', ...data }));
        game.on('memory-update', (data) => io.emit('memory-update', data));

        // Send initial state for the new game
        socket.emit('game-state', game.getGameSummary());
        
        // Start the game loop for the single game
        const result = await runGameLoop(game);
        handleGameResult(result);

      } catch (error) {
        console.error('Failed to start game:', error);
        socket.emit('error', { message: 'Failed to start game' });
      }
    });
  });

  const PORT = process.env.PORT || 3000;
  server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  }).on('error', (error) => {
    console.error('Server startup error:', error);
  });
}

/**
 * Runs the main game loop for a given game instance.
 * @param {Game} gameInstance - The game to run.
 * @returns {Promise<Object>} A promise that resolves with the game result data.
 */
async function runGameLoop(gameInstance) {
  try {
    while (!gameInstance.isGameOver()) {
      await gameInstance.processTurn();
      // A small delay to make the UI updates observable
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Construct a comprehensive result object
    const resultData = {
        winner: gameInstance.winner ? {
            name: gameInstance.winner.name,
            model: gameInstance.winner.model,
        } : null,
        players: gameInstance.agents.map(agent => ({
            name: agent.name,
            model: agent.model,
        })),
        log: gameInstance.gameLog, // Detailed log for historical view
        solution: gameInstance.solution,
        totalTurns: gameInstance.currentTurn,
        timestamp: new Date().toISOString(),
    };
    
    // Save to the persistent JSON file
    await GameResult.saveResults(resultData);

    return resultData; // Return the result for in-memory processing
  } catch (error) {
    console.error('Game loop error:', error);
    console.error('Game state:', {
      currentTurn: gameInstance.currentTurn,
      agents: gameInstance.agents?.map(a => a.name),
      solution: gameInstance.solution,
    });
    console.error(error.stack);
    // Return a partial result on error to avoid crashing the multi-game loop
    return { error: true, message: error.message };
  }
}
