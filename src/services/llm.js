import 'dotenv/config';
import Ajv from 'ajv';
import axios from 'axios';
import { Game } from '../models/Game.js';
import { logger } from '../utils/logger.js';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { CohereClient } from 'cohere-ai';
import { LoggingService } from './LoggingService.js';

// Add these near the top of the file with other imports
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// --- Configuration ---
const LLM_BACKEND = process.env.LLM_BACKEND || 'COHERE'; // Default to Cohere if not set
const ART_WRAPPER_URL = process.env.ART_WRAPPER_URL || 'http://localhost:5001';
const LLM_REQUEST_TIMEOUT = process.env.LLM_REQUEST_TIMEOUT || 60000; // 60 seconds timeout

logger.info(`Using LLM Backend: ${LLM_BACKEND}`);

/**
 * List of LLM models to be used for AI agents.
 * 
 * Each game will randomly assign these models to agents, allowing
 * for fair comparison of model capabilities in the Cluedo environment.
 * Models should be compatible with the Cohere API.
 * 
 * @type {Array<string>}
 */
export const MODEL_LIST = [
  'command-r-plus-04-2024',          
  'command-a-03-2025',     
  'c4ai-aya-expanse-32b',      
  'command-r-plus-04-2024',          
  'command-a-03-2025',     
  'c4ai-aya-expanse-32b'     
];

// Initialize JSON schema validator
const ajv = new Ajv();

// Define JSON schemas (Only accusationSchema is actively used for validation now)
/* // Removed suggestionSchema as validation is simpler now
const suggestionSchema = {
  type: "object",
  properties: {
    suspect: { type: "string" },
    weapon: { type: "string" },
    room: { type: "string" },
    reasoning: { type: "string" }
  },
  required: ["suspect", "weapon", "room", "reasoning"], 
  additionalProperties: false
};
*/

const accusationSchema = {
  type: "object",
  properties: {
    shouldAccuse: { type: "boolean" },
    accusation: {
      type: "object",
      properties: {
        suspect: { type: "string", nullable: true },
        weapon: { type: "string", nullable: true },
        room: { type: "string", nullable: true }
      },
      required: ["suspect", "weapon", "room"]
    },
    confidence: {
      type: "object",
      properties: {
        suspect: { type: "number" },
        weapon: { type: "number" },
        room: { type: "number" }
      },
      required: ["suspect", "weapon", "room"]
    },
    reasoning: { type: "string" }
  },
  required: ["shouldAccuse", "accusation", "confidence", "reasoning"]
};

// Replace with this custom JSON parser:
function extractJSON(response) {
  try {
    // Check if the response contains markdown code blocks
    const jsonMatch = response.match(/```(?:json)?\n([\s\S]*?)\n```/);
    if (jsonMatch) {
      logger.info('Found JSON wrapped in markdown code blocks, extracting content');
      return JSON.parse(jsonMatch[1]);
    }
    
    // Handle plain JSON
    return JSON.parse(response);
  } catch (error) {
    logger.error('JSON extraction failed:', { error: error.message, response });
    return null;
  }
}

// Update the safeParseJSON function:
function safeParseJSON(response, schema) {
  const parsed = extractJSON(response);
  if (!parsed) return { valid: false, error: 'Invalid JSON structure' };

  // Normalize confidence values
  if (schema === accusationSchema && parsed.confidence) {
    ['suspect', 'weapon', 'room'].forEach(field => {
      const value = parsed.confidence[field];
      
      // Handle string percentages
      if (typeof value === 'string') {
        const num = parseFloat(value.replace('%', ''));
        parsed.confidence[field] = isNaN(num) ? 0 : num;
      }
      
      // Convert to 0-1 scale if > 1
      if (typeof parsed.confidence[field] === 'number') {
        parsed.confidence[field] = parsed.confidence[field] > 1 
          ? parsed.confidence[field] / 100 
          : parsed.confidence[field];
      }
    });
  }

  const validate = ajv.compile(schema);
  if (!validate(parsed)) {
    console.error('Validation errors:', validate.errors);
    return { valid: false, error: validate.errors };
  }
  
  return { valid: true, data: parsed };
}

/**
 * @deprecated Use GameResult.saveResults instead
 * 
 * This redirection function is here for backward compatibility and will be removed in a future version.
 */
export async function saveGameResult(result) {
  console.warn('DEPRECATED: saveGameResult is deprecated. Use GameResult.saveResults instead');
  
  try {
    // Import GameResult dynamically to avoid circular dependencies
    const { GameResult } = await import('../models/GameResult.js');
    await GameResult.saveResults(result);
  } catch (error) {
    console.error('Error redirecting to GameResult.saveResults:', error);
    throw new Error('Failed to save game result. Please use GameResult.saveResults directly.');
  }
}

// --- Conditionally Initialize Cohere Client ---
let cohere = null;
if (LLM_BACKEND === 'COHERE') {
  if (!process.env.COHERE_API_KEY) {
    logger.warn('COHERE_API_KEY not found in environment variables. Cohere backend will likely fail.');
  }
  cohere = new CohereClient({
    token: process.env.COHERE_API_KEY,
    clientOptions: {
      timeoutInSeconds: 60 // Add a 60-second timeout
    }
  });
  logger.info('Cohere client initialized.');
}

export class LLMService {
  /**
   * Helper function to call the ART wrapper (Only used if LLM_BACKEND === 'ART')
   */
  static async _callArtWrapper(payload) {
    const endpoint = `${ART_WRAPPER_URL}/llm_request`;
    logger.debug(`Calling ART Wrapper at ${endpoint} for agent ${payload.agent_name}, task ${payload.task_type}`);
    try {
        const response = await axios.post(endpoint, payload, {
          timeout: LLM_REQUEST_TIMEOUT,
          headers: { 'Content-Type': 'application/json' }
        });
        logger.debug(`Received response from ART Wrapper: ${JSON.stringify(response.data)}`);
        if (response.status === 200 && response.data && response.data.request_id && response.data.content) {
            return { success: true, requestId: response.data.request_id, content: response.data.content };
        } else {
            logger.error(`Invalid response structure from ART wrapper: Status ${response.status}, Data: ${JSON.stringify(response.data)}`);
            return { success: false, error: `Invalid response structure from ART wrapper: ${response.status}` };
        }
    } catch (error) {
        logger.error(`Error calling ART wrapper at ${endpoint}: ${error.message}`, { error });
        let errorMessage = 'Failed to connect to ART wrapper';
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            errorMessage = `ART wrapper error: ${error.response.status} - ${JSON.stringify(error.response.data)}`;
        } else if (error.request) {
            // The request was made but no response was received
            errorMessage = `No response received from ART wrapper at ${endpoint}`;
        } else if (error.code === 'ECONNABORTED') {
            errorMessage = `ART wrapper request timed out after ${LLM_REQUEST_TIMEOUT / 1000}s`;
        } else {
            // Something happened in setting up the request that triggered an Error
            errorMessage = `Error setting up request to ART wrapper: ${error.message}`;
        }
        return { success: false, error: errorMessage };
    }
  }

  /**
   * Generates a strategic suggestion for an agent during their turn.
   */
  static async makeSuggestion(agent, gameState) {
    const startTime = Date.now();
    const taskType = 'suggestion';
    const loggingPayload = { // Common structure for logging
        type: taskType,
      agent: agent.name,
        model: agent.model, // Model used (Cohere or ART base model)
      input: {},
      output: null,
      error: null,
      parsedOutput: null,
      validationStatus: 'pending'
    };

    try {
      const memoryState = await agent.memory.formatMemoryForLLM();
        const prompt = `Analyze the game state and make a strategic suggestion:
Known cards held: ${Array.from(agent.cards).join(', ')}
Current turn number: ${gameState.currentTurn}
Your current location: ${agent.location} (You must suggest this room)
Available suspects (excluding yourself, ${agent.name}): ${gameState.availableSuspects.filter(s => s !== agent.name).join(', ')}
Available weapons: ${gameState.availableWeapons.join(', ')}
Available rooms (you must choose ${agent.location}): ${agent.location}

Your memory and deductions:
Known Information: ${JSON.stringify(memoryState.knownInformation, null, 2)}
Current Deductions: ${memoryState.currentDeductions}
Recent Turn History:
${memoryState.turnHistory.join('\n')}

Make a strategic suggestion considering:
1. Your known cards and deductions.
2. Previous suggestions and their outcomes (from Turn History).
3. Information revealed by other players.
4. Your current room (${agent.location}) - you MUST suggest this room.
5. Choose a suspect (not yourself) and a weapon that seem most likely based on your deductions, or that would gather the most information.

Respond ONLY with a JSON object in the following format.
IMPORTANT: Do NOT use markdown code blocks (\`\`\`json) in your response - just return the JSON object directly.

{
  "suspect": "string (must be an available suspect)",
  "weapon": "string (must be an available weapon)",
  "room": "string (must be your current room: ${agent.location})",
  "reasoning": "string (explain your strategy and deduction process briefly)"
}`;
        loggingPayload.input = { prompt, /* other relevant input state */ };

        let responseText = '';
        let requestId = null; // Only relevant for ART
        let llmResponse;

        if (LLM_BACKEND === 'ART') {
            logger.debug(`[ART] Calling wrapper for suggestion...`);
            const payload = { agent_name: agent.name, turn_number: gameState.currentTurn, task_type: taskType, prompt };
            const artResponse = await LLMService._callArtWrapper(payload);
            if (!artResponse.success) throw new Error(artResponse.error || 'ART wrapper call failed');
            responseText = artResponse.content;
            requestId = artResponse.requestId;
            loggingPayload.output = responseText; // Log raw ART output
        } else { // Default to COHERE
            logger.debug(`[Cohere] Calling API for suggestion...`);
            if (!cohere) throw new Error('Cohere client not initialized');
            const supportsJsonResponseFormat = agent.model.startsWith('command-r');
            const apiParams = { model: agent.model, message: prompt, temperature: 0.1 };
            if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };
            
            llmResponse = await cohere.chat(apiParams);
            responseText = llmResponse.text;
            loggingPayload.output = responseText; // Log raw Cohere output
        }

        // --- Parsing and Validation (Common Logic) ---
        let parsedResult = extractJSON(responseText);
        if (!parsedResult || typeof parsedResult !== 'object') {
            loggingPayload.validationStatus = 'failed_parsing';
            throw new Error('Failed to parse JSON response from LLM');
        }
        loggingPayload.parsedOutput = parsedResult;

        if (!parsedResult.suspect || !parsedResult.weapon || !parsedResult.room || !parsedResult.reasoning) {
          loggingPayload.validationStatus = 'failed_validation';
            throw new Error('LLM response missing required fields');
      }
      if (parsedResult.room !== agent.location) {
            logger.warn(`[${LLM_BACKEND}] ${agent.name} suggestion: Room mismatch. Agent in ${agent.location}, suggested ${parsedResult.room}. Overriding.`);
           parsedResult.room = agent.location;
           loggingPayload.validationStatus = 'corrected_room';
      } else {
           loggingPayload.validationStatus = 'passed';
      }
        if (!gameState.availableSuspects.includes(parsedResult.suspect) || !gameState.availableWeapons.includes(parsedResult.weapon)) {
            logger.warn(`[${LLM_BACKEND}] ${agent.name} suggestion: Invalid suspect or weapon suggested. S:${parsedResult.suspect}, W:${parsedResult.weapon}`);
            // TODO: Decide how to handle - fallback or error? For now, allow but warn.
        }

        // --- Logging (Conditional) ---
        if (LLM_BACKEND === 'COHERE') {
            await LoggingService.logLLMInteraction(loggingPayload); // Log Cohere interaction
        }
        
        logger.info(`[${LLM_BACKEND}] Suggestion for ${agent.name} took ${Date.now() - startTime}ms`);
        
        // Return result, include requestId only if using ART
        const finalResult = { 
            suspect: parsedResult.suspect, 
            weapon: parsedResult.weapon, 
            room: parsedResult.room, 
            reasoning: parsedResult.reasoning 
        };
        if (LLM_BACKEND === 'ART') {
            finalResult.requestId = requestId;
        }
        return finalResult;

    } catch (error) {
        logger.error(`[${LLM_BACKEND}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
      loggingPayload.error = error.message;
      loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

        // --- Logging (Conditional on Error) ---
        if (LLM_BACKEND === 'COHERE') {
            await LoggingService.logLLMInteraction(loggingPayload); // Log failed Cohere interaction
        }

        // Consistent fallback structure, no requestId on error
        return {
            suspect: gameState.availableSuspects ? gameState.availableSuspects[0] : 'Miss Scarlet',
            weapon: gameState.availableWeapons ? gameState.availableWeapons[0] : 'Candlestick',
            room: agent.location || 'Lounge',
            reasoning: `Error occurred during ${taskType} via ${LLM_BACKEND}: ${error.message}, using fallback.`,
            error: error.message || `Unknown error during ${taskType}`
        };
    }
  }
  

  /**
   * Updates the agent's memory based on turn events.
   */
  static async updateMemory(agent, memory, turnEvents) {
      const startTime = Date.now();
      const taskType = 'memory_update';
      const loggingPayload = { /* ... common logging structure ... */ }; // Initialize if logging Cohere

    if (!turnEvents || turnEvents.length === 0) {
          // Consistent return structure for skipped update
          return { deducedCards: [], summary: '(Memory update skipped, no events)' }; 
      }

      try {
          const memoryState = await memory.formatMemoryForLLM();
          const prompt = `Analyze the following events from the last turn and update your memory and deductions. Identify any newly deduced cards (suspect, weapon, or room) that are definitively NOT part of the solution based ONLY on these events and your existing knowledge.

Your current knowledge:
Known cards held: ${Array.from(agent.cards).join(', ')}
Known Information: ${JSON.stringify(memoryState.knownInformation, null, 2)}
Current Deductions: ${memoryState.currentDeductions}

Events from last turn:
${turnEvents.join('\n')}

Based *only* on the information above, update your deductions and provide a concise summary of the key learnings from these events.

Respond ONLY with a JSON object in the following format. Provide an empty list if no new cards were deduced.
IMPORTANT: Do NOT use markdown code blocks (\`\`\`json) in your response - just return the JSON object directly.

{
  "newlyDeducedCards": ["string"], 
  "reasoning": "string (briefly explain how you deduced the cards or why none could be deduced)",
  "memorySummary": "string (concise summary of updated memory/key takeaways from events)"
}`;
          loggingPayload.input = { prompt, /* ... */ };

          let responseText = '';
          let requestId = null;
          let llmResponse;

          if (LLM_BACKEND === 'ART') {
              logger.debug(`[ART] Calling wrapper for ${taskType}...`);
              const payload = { agent_name: agent.name, turn_number: agent.game?.currentTurn ?? -1, task_type: taskType, prompt };
              const artResponse = await LLMService._callArtWrapper(payload);
              if (!artResponse.success) throw new Error('ART wrapper call failed for memory update');
              responseText = artResponse.content;
              requestId = artResponse.requestId;
              loggingPayload.output = responseText; // Log raw ART output
          } else { // Default to COHERE
              logger.debug(`[Cohere] Calling API for ${taskType}...`);
              if (!cohere) throw new Error('Cohere client not initialized');
        const supportsJsonResponseFormat = agent.model.startsWith('command-r');
              const apiParams = { model: agent.model, message: prompt, temperature: 0.1 };
              if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };
              
              llmResponse = await cohere.chat(apiParams);
              responseText = llmResponse.text;
              loggingPayload.output = responseText; // Log raw Cohere output
          }

          // --- Parsing and Validation (Common Logic) ---
          const parsedResult = extractJSON(responseText);
          if (!parsedResult || typeof parsedResult !== 'object') {
              loggingPayload.validationStatus = 'failed_parsing';
              throw new Error(`Failed to parse JSON response from LLM for ${taskType}`);
          }
          loggingPayload.parsedOutput = parsedResult;

          const deducedCards = parsedResult.newlyDeducedCards || [];
          const summary = parsedResult.memorySummary || parsedResult.reasoning || '(No summary provided)';
          const reasoning = parsedResult.reasoning || '(No reasoning provided)';

          if (!Array.isArray(deducedCards)) {
              loggingPayload.validationStatus = 'failed_validation';
              throw new Error('Invalid format: newlyDeducedCards should be an array.');
          }
          loggingPayload.validationStatus = 'passed';
          
          // Update memory object (Common Logic - assumes memory.update exists)
          if (memory.update) {
              memory.update(summary, deducedCards, reasoning);
          } else {
              logger.warn(`[${LLM_BACKEND}] ${agent.name}: Memory object does not have an update method.`);
          }

          // --- Logging (Conditional) ---
          if (LLM_BACKEND === 'COHERE') {
              await LoggingService.logLLMInteraction(loggingPayload); // Log Cohere interaction
          }

          logger.info(`[${LLM_BACKEND}] ${taskType} for ${agent.name} took ${Date.now() - startTime}ms`);

          // Consistent return structure, add requestId only for ART
          const finalResult = { deducedCards, summary };
          if (LLM_BACKEND === 'ART') {
              finalResult.requestId = requestId;
          }
          return finalResult;

      } catch (error) {
          logger.error(`[${LLM_BACKEND}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
          loggingPayload.error = error.message;
          loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

          // --- Logging (Conditional on Error) ---
          if (LLM_BACKEND === 'COHERE') {
              await LoggingService.logLLMInteraction(loggingPayload); // Log failed Cohere interaction
          }
          
          // Consistent fallback structure
          return { 
              deducedCards: [], 
              summary: `(Error during ${taskType} via ${LLM_BACKEND}: ${error.message})`, 
              error: error.message || `Unknown error during ${taskType}`
          };
    }
  }

  /**
   * Evaluates a challenge and decides which card to show.
   */
  static async evaluateChallenge(agent, suggestion, cards) {
      const startTime = Date.now();
      const taskType = 'evaluate_challenge';
      const loggingPayload = { /* ... common logging structure ... */ }; // Initialize if logging Cohere

      if (!cards || cards.length === 0) {
          return { cardToShow: null, reasoning: "No matching cards to show" };
      }

      try {
          const memoryState = await agent.memory.formatMemoryForLLM();
          const prompt = `You received a suggestion: ${suggestion.suspect}, ${suggestion.weapon}, ${suggestion.room}.\nYou hold the following matching card(s): ${cards.join(', ')}.\n\nYour current knowledge:\nKnown cards held: ${Array.from(agent.cards).join(', ')} || 'None'\nKnown Information: ${JSON.stringify(memoryState.knownInformation, null, 2)}\nCurrent Deductions: ${memoryState.currentDeductions}\n\nChoose ONE card from your matching cards (${cards.join(', ')}) to show to the suggester. Consider which card reveals the least about your overall hand and deductions, while still disproving the suggestion.\n\nRespond ONLY with a JSON object in the following format.\nIMPORTANT: Do NOT use markdown code blocks (\`\`\`json) in your response - just return the JSON object directly.\n\n{\n  "cardToShow": "string (must be one of: ${cards.join(', ')})",\n  "reasoning": "string (briefly explain your choice)"\n}`;
          loggingPayload.input = { prompt, /* ... */ };

          let responseText = '';
          let requestId = null;
          let llmResponse;

          if (LLM_BACKEND === 'ART') {
              logger.debug(`[ART] Calling wrapper for ${taskType}...`);
              const payload = { agent_name: agent.name, turn_number: agent.game?.currentTurn ?? -1, task_type: taskType, prompt };
              const artResponse = await LLMService._callArtWrapper(payload);
              if (!artResponse.success) throw new Error('ART wrapper call failed for challenge evaluation');
              responseText = artResponse.content;
              requestId = artResponse.requestId;
              loggingPayload.output = responseText; // Log raw ART output
          } else { // Default to COHERE
              logger.debug(`[Cohere] Calling API for ${taskType}...`);
              if (!cohere) throw new Error('Cohere client not initialized');
              const supportsJsonResponseFormat = agent.model.startsWith('command-r');
              const apiParams = { model: agent.model, message: prompt, temperature: 0.1 };
              if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };
              
              llmResponse = await cohere.chat(apiParams);
              responseText = llmResponse.text;
              loggingPayload.output = responseText; // Log raw Cohere output
          }

          // --- Parsing and Validation (Common Logic) ---
          let parsedResult = extractJSON(responseText);
          if (!parsedResult || typeof parsedResult !== 'object' || !parsedResult.cardToShow) {
              loggingPayload.validationStatus = 'failed_parsing';
              throw new Error(`Failed to parse JSON response or missing cardToShow for ${taskType}`);
            }
            loggingPayload.parsedOutput = parsedResult;

          if (!cards.includes(parsedResult.cardToShow)) {
              logger.warn(`[${LLM_BACKEND}] ${agent.name} ${taskType}: LLM chose card (${parsedResult.cardToShow}) not in matching set (${cards.join(', ')}). Falling back.`);
              parsedResult.cardToShow = cards[0]; 
              parsedResult.reasoning += " (LLM response invalid, showing first available card)";
            loggingPayload.validationStatus = 'corrected_card';
        } else {
            loggingPayload.validationStatus = 'passed';
        }

          const reasoning = parsedResult.reasoning || '(No reasoning provided)';

          // --- Logging (Conditional) ---
          if (LLM_BACKEND === 'COHERE') {
              await LoggingService.logLLMInteraction(loggingPayload); // Log Cohere interaction
          }

          logger.info(`[${LLM_BACKEND}] ${taskType} for ${agent.name} took ${Date.now() - startTime}ms`);

          // Consistent return structure, add requestId only for ART
          const finalResult = { cardToShow: parsedResult.cardToShow, reasoning };
          if (LLM_BACKEND === 'ART') {
              finalResult.requestId = requestId;
          }
          return finalResult;

    } catch (error) {
          logger.error(`[${LLM_BACKEND}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
        loggingPayload.error = error.message;
        loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

          // --- Logging (Conditional on Error) ---
          if (LLM_BACKEND === 'COHERE') {
              await LoggingService.logLLMInteraction(loggingPayload); // Log failed Cohere interaction
          }
          
          // Consistent fallback structure
          const fallbackCard = cards[0] || null;
          return {
              cardToShow: fallbackCard,
              reasoning: `Error during ${taskType} via ${LLM_BACKEND}: ${error.message}. Falling back to showing ${fallbackCard || 'nothing'}.`,
              error: error.message || `Unknown error during ${taskType}`
          };
    }
  }

  /**
   * Decides whether the agent should make an accusation.
   */
  static async considerAccusation(agent, gameState) {
      const startTime = Date.now();
      const taskType = 'consider_accusation';
      const loggingPayload = { /* ... common logging structure ... */ }; // Initialize if logging Cohere

    try {
      const memoryState = await agent.memory.formatMemoryForLLM();
          const prompt = `Based on your complete knowledge, decide if you are confident enough to make a final accusation to win the game. 

Your knowledge:
Known cards held: ${Array.from(agent.cards).join(', ')} || 'None'
Known Information: ${JSON.stringify(memoryState.knownInformation, null, 2)}
Current Deductions: ${memoryState.currentDeductions}
Turn History Snapshot:
${memoryState.turnHistory.slice(-10).join('\n')}

Current turn number: ${gameState.currentTurn}

Consider the certainty of your deductions for the suspect, weapon, and room.

Respond ONLY with a JSON object in the following format.
If shouldAccuse is true, provide your deduced solution and confidence (0.0-1.0).
If shouldAccuse is false, provide null for accusation components and 0 for confidence.
IMPORTANT: Do NOT use markdown code blocks (\`\`\`json) in your response - just return the JSON object directly.

{
  "shouldAccuse": boolean,
  "accusation": {
    "suspect": "string | null (your deduced suspect or null)",
    "weapon": "string | null (your deduced weapon or null)",
    "room": "string | null (your deduced room or null)"
  },
  "confidence": {
    "suspect": number (0.0-1.0),
    "weapon": number (0.0-1.0),
    "room": number (0.0-1.0)
  },
  "reasoning": "string (explain your decision and confidence level)"
}`;
          loggingPayload.input = { prompt, /* ... */ };

          let responseText = '';
          let requestId = null;
          let llmResponse;

          if (LLM_BACKEND === 'ART') {
              logger.debug(`[ART] Calling wrapper for ${taskType}...`);
              const payload = { agent_name: agent.name, turn_number: gameState.currentTurn, task_type: taskType, prompt };
              const artResponse = await LLMService._callArtWrapper(payload);
              if (!artResponse.success) throw new Error('ART wrapper call failed for accusation consideration');
              responseText = artResponse.content;
              requestId = artResponse.requestId;
              loggingPayload.output = responseText; // Log raw ART output
          } else { // Default to COHERE
              logger.debug(`[Cohere] Calling API for ${taskType}...`);
              if (!cohere) throw new Error('Cohere client not initialized');
      const supportsJsonResponseFormat = agent.model.startsWith('command-r');
              const apiParams = { model: agent.model, message: prompt, temperature: 0.1 };
              if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };
              
              llmResponse = await cohere.chat(apiParams);
              responseText = llmResponse.text;
              loggingPayload.output = responseText; // Log raw Cohere output
          }

          // --- Parsing and Validation (Common Logic using schema) ---
          const validationResult = safeParseJSON(responseText, accusationSchema);
          if (!validationResult.valid) {
              loggingPayload.validationStatus = 'failed_validation';
              logger.warn(`[${LLM_BACKEND}] ${agent.name} ${taskType}: Failed validation: ${JSON.stringify(validationResult.error)}. Response: ${responseText}`);
              throw new Error(`Accusation response failed validation: ${JSON.stringify(validationResult.error)}`);
          }
          const parsedResult = validationResult.data;
          loggingPayload.parsedOutput = parsedResult;
          loggingPayload.validationStatus = 'passed';

          // Additional check
          if (parsedResult.shouldAccuse && 
              (!parsedResult.accusation.suspect || !parsedResult.accusation.weapon || !parsedResult.accusation.room)) {
              logger.warn(`[${LLM_BACKEND}] ${agent.name} ${taskType}: shouldAccuse is true but accusation details missing/null. Overriding to false.`);
              parsedResult.shouldAccuse = false;
              parsedResult.reasoning += " (Invalid accusation details provided, overriding shouldAccuse to false)";
              loggingPayload.validationStatus = 'corrected_logic';
          }

          // --- Logging (Conditional) ---
          if (LLM_BACKEND === 'COHERE') {
              await LoggingService.logLLMInteraction(loggingPayload); // Log Cohere interaction
          }

          logger.info(`[${LLM_BACKEND}] ${taskType} for ${agent.name} took ${Date.now() - startTime}ms`);

          // Consistent return structure, add requestId only for ART
          const finalResult = { 
              shouldAccuse: parsedResult.shouldAccuse, 
              accusation: parsedResult.accusation, 
              confidence: parsedResult.confidence, 
              reasoning: parsedResult.reasoning 
          };
          if (LLM_BACKEND === 'ART') {
              finalResult.requestId = requestId;
          }
          return finalResult;

    } catch (error) {
          logger.error(`[${LLM_BACKEND}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
      loggingPayload.error = error.message;
      loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

          // --- Logging (Conditional on Error) ---
          if (LLM_BACKEND === 'COHERE') {
              await LoggingService.logLLMInteraction(loggingPayload); // Log failed Cohere interaction
          }

          // Consistent fallback structure
      return {
        shouldAccuse: false,
        accusation: { suspect: null, weapon: null, room: null },
        confidence: { suspect: 0, weapon: 0, room: 0 },
              reasoning: `Error during ${taskType} via ${LLM_BACKEND}: ${error.message}. Defaulting to not accuse.`,
              error: error.message || `Unknown error during ${taskType}`
      };
    }
  }
} 