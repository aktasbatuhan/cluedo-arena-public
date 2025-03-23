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
    try {
      const resultPath = path.join(__dirname, '../../game_results.json');
      
      // Create directory if it doesn't exist
      const dir = path.dirname(resultPath);
      try {
        await mkdir(dir, { recursive: true });
      } catch (dirError) {
        // Ignore if directory already exists
        if (dirError.code !== 'EEXIST') {
          console.error('Failed to create directory:', dirError);
          // Continue anyway
        }
      }
      
      let existing = [];
      try {
        const data = await readFile(resultPath, 'utf8');
        existing = JSON.parse(data);
      } catch (readError) {
        console.error('Reading existing results failed, starting new file:', readError.message);
        existing = [];
      }
      
      const updated = [...existing, results];
      
      try {
        await writeFile(resultPath, JSON.stringify(updated, null, 2));
        console.log('Game results saved successfully to:', resultPath);
      } catch (writeError) {
        console.error('Error writing game results:', writeError);
        
        // Attempt to save backup copy
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
      console.error('Error in GameResult.saveResults:', error);
      throw error;
    }
  }
} 