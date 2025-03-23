import { writeFile, readFile } from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export class LoggingService {
  static async logLLMInteraction(data) {
    const logPath = path.join(__dirname, '../../llm_interactions.json');
    
    try {
      // Read existing logs
      let logs = [];
      try {
        const existingData = await readFile(logPath, 'utf8');
        logs = JSON.parse(existingData);
      } catch (e) {
        // File doesn't exist yet, start with empty array
        logs = [];
      }

      // Add new log entry
      logs.push({
        ...data,
        timestamp: new Date().toISOString()
      });

      // Write updated logs
      await writeFile(logPath, JSON.stringify(logs, null, 2));
    } catch (error) {
      console.error('Failed to log LLM interaction:', error);
    }
  }
} 