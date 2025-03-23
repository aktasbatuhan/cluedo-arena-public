import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { Game } from './models/Game.js';
import { runGame } from './gameLogic.js';
import { GameResult } from './models/GameResult.js';
import { saveGameResult } from './services/llm.js';
import { parseArgs } from 'node:util';

let game = null;

export async function startServer() {
  // Parse command line arguments
  const { values } = parseArgs({
    options: {
      games: { type: 'string' }, // --games=10
      mode: { type: 'string' }   // --mode=batch
    }
  });

  // If batch mode is specified, run multiple games
  if (values.mode === 'batch' && values.games) {
    const numGames = parseInt(values.games);
    console.log(`Starting batch mode: Running ${numGames} games...`);
    
    for (let i = 0; i < numGames; i++) {
      console.log(`\n=== Starting Game ${i + 1}/${numGames} ===\n`);
      const game = new Game('spectate');
      await game.initialize();
      await runGameLoop(game);
      
      // Clear all agent memories before next game
      if (game.agents) {
        for (const agent of game.agents) {
          await agent.memory.reset();
        }
      }
      
      // Small delay between games
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log('\nBatch mode completed. All games finished.');
    process.exit(0);
    return;
  }

  const app = express();
  const server = createServer(app);
  const io = new Server(server);
  
  app.use(express.static('public'));
  
  io.on('connection', (socket) => {
    socket.on('game-mode', async (mode) => {
      try {
        // Always create a new game when mode is selected
        game = new Game(mode, io);
        await game.initialize();
        
        // Set up event listeners for UI updates
        game.on('suggestion', (data) => {
          io.emit('game-event', {
            type: 'SUGGESTION',
            timestamp: new Date(),
            agent: data.agent,
            suggestion: data.suggestion,
            reasoning: data.reasoning,
            message: `${data.agent} suggests ${data.suggestion.suspect} in the ${data.suggestion.room} with the ${data.suggestion.weapon}`
          });
        });

        game.on('challenge', (data) => {
          io.emit('game-event', {
            type: 'CHALLENGE',
            timestamp: new Date(),
            agent: data.agent,
            cardShown: data.cardShown,
            message: `${data.agent} showed ${data.cardShown} to disprove the suggestion`
          });
        });

        game.on('accusation', async (data) => {
          if (game?.mode === 'play' && game.humanPlayer) {
            const isCorrectAccusation = await game.processHumanAccusation(data.accusation);
            if (isCorrectAccusation) {
              const winner = game.gameState.players.find(player => player.color === game.currentPlayer.color);
              io.emit('GAME_EVENT', { 
                type: 'GAME_OVER',
                winner: { name: winner.name },
                timestamp: new Date().toISOString()
              });
              
              // Save game result
              await saveGameResult({
                winner: winner.name,
                players: game.gameState.players,
                accusation: data.accusation,
                timestamp: new Date().toISOString()
              });

              game.resetGame();
            } else {
              // Handle incorrect accusation
              game.gameState.players = game.gameState.players.filter(p => p.color !== game.currentPlayer.color);
              io.emit('GAME_EVENT', {
                type: 'ACCUSATION_ATTEMPT',
                agent: game.currentPlayer.name,
                accusation: data.accusation,
                eliminated: true,
                timestamp: new Date().toISOString()
              });
              
              // Check if any players left
              if (game.gameState.players.length === 0) {
                await saveGameResult({
                  winner: 'No winner',
                  players: game.originalPlayers,
                  timestamp: new Date().toISOString()
                });
                game.resetGame();
              }
            }
          }
        });

        // Add turn update events
        game.on('turn-start', (data) => {
          io.emit('game-event', {
            type: 'TURN_START',
            timestamp: new Date(),
            agent: data.agent,
            message: `${data.agent}'s turn`
          });
        });

        // Send initial state
        socket.emit('game-state', game.getGameSummary());
        
        if (mode === 'spectate') {
          console.log('Starting AI-only game...');
          runGameLoop(game);
        }
      } catch (error) {
        console.error('Failed to start game:', error);
        socket.emit('error', { message: 'Failed to start game' });
      }
    });

    socket.on('player-action', async (action) => {
      if (game?.mode === 'play' && game.humanPlayer) {
        await game.processHumanAction(action);
      }
    });
  });

  // Add results route
  app.get('/results', (req, res) => {
    const fs = require('fs');
    const path = require('path');
    const resultPath = path.join(__dirname, '../game_results.json');
    
    try {
      const results = JSON.parse(fs.readFileSync(resultPath));
      res.send(`
        <html>
          <head>
            <title>Game Results</title>
            <style>
              table { margin-bottom: 20px; }
              .stats-section { margin-bottom: 30px; }
              .game-log {
                font-family: monospace;
                white-space: pre-wrap;
                background: #f5f5f5;
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
              }
            </style>
          </head>
          <body>
            <h1>Game Statistics</h1>
            
            <div class="stats-section">
              <h2>Model Performance</h2>
              <div id="model-stats"></div>
            </div>

            <div class="stats-section">
              <h2>Detailed Game Logs</h2>
              <div id="game-logs"></div>
            </div>

            <script>
              const results = ${JSON.stringify(results)};
              
              // Model stats calculation
              const modelStats = {};
              results.forEach(game => {
                game.participants.forEach(participant => {
                  if (!modelStats[participant.model]) {
                    modelStats[participant.model] = {
                      games: 0,
                      wins: 0,
                      totalTurns: 0
                    };
                  }
                  modelStats[participant.model].games++;
                  modelStats[participant.model].totalTurns += game.turns;
                  
                  if (game.winner && participant.name === game.winner.name) {
                    modelStats[participant.model].wins++;
                  }
                });
              });

              let modelStatsHtml = '<table border="1">';
              modelStatsHtml += '<tr><th>Model</th><th>Games Played</th><th>Wins</th><th>Win Rate</th><th>Avg Turns</th></tr>';
              
              Object.entries(modelStats).forEach(([model, stats]) => {
                modelStatsHtml += \`<tr>
                  <td>\${model}</td>
                  <td>\${stats.games}</td>
                  <td>\${stats.wins}</td>
                  <td>\${(stats.wins/stats.games*100).toFixed(1)}%</td>
                  <td>\${(stats.totalTurns/stats.games).toFixed(1)}</td>
                </tr>\`;
              });
              
              modelStatsHtml += '</table>';

              // Add detailed logs
              const gameLogs = results.map((game, index) => {
                return \`
                  <details>
                    <summary>Game \${index + 1} - \${game.winner ? game.winner.name : 'No winner'}</summary>
                    <div class="game-log">
                      \${game.detailedLog.join('\n')}
                    </div>
                  </details>
                \`;
              }).join('');
              
              document.getElementById('game-logs').innerHTML = gameLogs;
            </script>
          </body>
        </html>
      `);
    } catch (e) {
      res.status(500).send('Error loading results');
    }
  });

  // Remove duplicate server startup
  return new Promise((resolve, reject) => {
    const PORT = process.env.PORT || 3000;
    server.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
      resolve();
    }).on('error', error => {
      if (error.code === 'EADDRINUSE') {
        console.error(`Port ${PORT} is already in use. Please either:
1. Close other running instances
2. Use a different port: PORT=4000 npm start
3. Kill existing process: lsof -i tcp:${PORT} | awk 'NR!=1 {print $2}' | xargs kill -9`);
      }
      reject(error);
    });
  });
}

async function runGameLoop(game) {
  try {
    console.log('Starting game loop...');
    const missedOpportunities = new Map();
    
    while (!game.isGameOver()) {
      const currentTurn = game.currentTurn;
      console.log(`\nProcessing turn ${currentTurn}`);
      
      const turnResult = await game.processTurn();
      
      if (turnResult?.suggestion) {
        const currentAgent = game.agents.find(a => a.name === turnResult.agent);
        
        if (currentAgent) {
          console.log('Checking suggestion:', {
            suggestion: turnResult.suggestion,
            solution: game.solution
          });
          
          if (turnResult.suggestion.suspect === game.solution.suspect &&
              turnResult.suggestion.weapon === game.solution.weapon &&
              turnResult.suggestion.room === game.solution.room) {
            
            console.log(`Found missed opportunity by ${currentAgent.name}`);
            
            const playerMisses = missedOpportunities.get(currentAgent.name) || [];
            playerMisses.push({
              turn: currentTurn,
              suggestion: turnResult.suggestion,
              confidence: turnResult.accusationDecision?.confidence || null,
              reasoning: turnResult.accusationDecision?.reasoning || null
            });
            missedOpportunities.set(currentAgent.name, playerMisses);
          }
        }
      }
      
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      if (game.winner || game.currentTurn >= game.maxTurns) {
        console.log('Game ending condition met');
        break;
      }
    }
    
    // Log missed opportunities before saving
    console.log('Missed Opportunities:', Object.fromEntries(missedOpportunities));
    
    console.log('Game loop completed');
    if (game.winner) {
      console.log(`Winner: ${game.winner.name}`);
      await saveGameResult({
        winner: {
          name: game.winner.name,
          model: game.winner.model
        },
        players: game.agents?.map(agent => ({
          name: agent.name,
          model: agent.model,
          missedOpportunities: missedOpportunities.get(agent.name) || []
        })) || [],
        timestamp: new Date().toISOString(),
        solution: game.solution,
        totalTurns: game.currentTurn
      });
    } else {
      console.log('Game ended without a winner');
      await saveGameResult({
        winner: 'No winner',
        players: game.agents?.map(agent => ({
          name: agent.name,
          model: agent.model,
          missedOpportunities: missedOpportunities.get(agent.name) || []
        })) || [],
        timestamp: new Date().toISOString(),
        solution: game.solution,
        totalTurns: game.currentTurn
      });
    }
    
  } catch (error) {
    console.error('Game loop error:', error);
    console.error('Game state:', {
      currentTurn: game.currentTurn,
      agents: game.agents?.map(a => a.name),
      solution: game.solution,
      missedOpportunities: Object.fromEntries(missedOpportunities)
    });
    console.error(error.stack);
  }
} 