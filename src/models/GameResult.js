import { writeFile, readFile, mkdir } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { Agent } from './Agent.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export class GameResult {
  constructor(game) {
    this.timestamp = new Date().toISOString();
    this.solution = game.solution;
    this.winner = game.winner ? {
      name: game.winner.name,
      model: game.winner.model
    } : null;
    this.turns = game.currentTurn;
    this.accusations = game.accusations;
    
    this.agents = game.agents.map(a => {
      return {
        name: a.name,
        model: a.model,
        hasLost: a.hasLost
      };
    });
  }
  

  toJSON() {
    return {
      timestamp: this.timestamp,
      solution: this.solution,
      winner: this.winner,
      totalTurns: this.turns,
      accusations: this.accusations,
      players: this.agents
    };
  }
  
  static async saveResults(results) {
    console.log('[GameResult.saveResults] Starting save process...');
    const resultPath = path.join(__dirname, '../../game_results.json');
    console.log(`[GameResult.saveResults] Determined result path: ${resultPath}`);
    try {
      const dir = path.dirname(resultPath);
      try {
        console.log(`[GameResult.saveResults] Ensuring directory exists: ${dir}`);
        await mkdir(dir, { recursive: true });
      } catch (dirError) {
        if (dirError.code !== 'EEXIST') {
          console.error('[GameResult.saveResults] Failed to create directory:', dirError);
        }
      }

      let existing = [];
      try {
        console.log('[GameResult.saveResults] Reading existing results file (if any)...');
        const data = await readFile(resultPath, 'utf8');
        existing = JSON.parse(data);
        console.log(`[GameResult.saveResults] Successfully read and parsed ${existing.length} existing results.`);
      } catch (readError) {
        if (readError.code === 'ENOENT') {
          console.log('[GameResult.saveResults] No existing results file found. Starting new file.');
        } else {
          console.error('[GameResult.saveResults] Reading existing results failed, starting new file:', readError.message);
        }
        existing = [];
      }

      const resultsToAppend = Array.isArray(results) ? results : [results];
      console.log(`[GameResult.saveResults] Appending ${resultsToAppend.length} new result(s).`);
      const updated = [...existing, ...resultsToAppend];

      try {
        console.log(`[GameResult.saveResults] Writing ${updated.length} total results to file...`);
        await writeFile(resultPath, JSON.stringify(updated, null, 2), { encoding: 'utf8' });
        console.log('[GameResult.saveResults] Game results saved successfully.');
      } catch (writeError) {
        console.error('[GameResult.saveResults] Error WRITING game results file:', writeError);
        try {
          const backupPath = `${resultPath}.backup-${Date.now()}`;
          await writeFile(backupPath, JSON.stringify(updated));
          console.error(`Created backup at: ${backupPath}`);
        } catch (backupError) {
          console.error('Failed to create backup:', backupError);
        }
        throw writeError;
      }
    } catch (error) {
      console.error('[GameResult.saveResults] Overall error in saveResults:', error);
      throw error;
    }
  }
} 