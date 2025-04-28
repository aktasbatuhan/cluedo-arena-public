import 'dotenv/config';
import Ajv from 'ajv';
import axios from 'axios';
import { Game } from '../models/Game.js';
import { logger } from '../utils/logger.js';
import fs from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';
import { CohereClient } from 'cohere-ai';
import { LoggingService } from './LoggingService.js';
import { OpenAI } from 'openai';       // Import OpenAI SDK
import dotenv from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

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
  'command-a-03-2025',          
  'command-a-03-2025',     
  'command-a-03-2025',      
  'command-a-03-2025',          
  'command-a-03-2025',     
  'command-a-03-2025'     
];

// Initialize JSON schema validator
const ajv = new Ajv();

// Define JSON schemas (Only accusationSchema is actively used for validation now)
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
    reasoning: { type: "string" }
  },
  required: ["shouldAccuse", "accusation", "reasoning"]
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

// --- Client Initialization ---

// Existing Cohere Client (Example)
const cohereClient = new CohereClient({
  token: process.env.CO_API_KEY,
});

// NEW: OpenRouter Client (using OpenAI SDK)
const openRouterClient = process.env.OPENROUTER_API_KEY ? new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
}) : null; // Initialize only if key exists

const YOUR_SITE_URL = process.env.YOUR_SITE_URL || "http://localhost:3000"; // Optional: Get from .env or default
const YOUR_SITE_NAME = process.env.YOUR_SITE_NAME || "Cluedo Arena";      // Optional: Get from .env or default


// --- Helper Function to Get Client/Backend ---
// (You'll need a way to decide which client to use)
// Option A: Based on agent.model string prefix
function getBackendConfig(agentModel) {
  if (agentModel?.startsWith('openai/') || agentModel?.startsWith('google/') || agentModel?.startsWith('mistralai/')) { // Add other OpenRouter compatible prefixes
      if (!openRouterClient) {
        throw new Error("OPENROUTER_API_KEY is not configured, but an OpenRouter model was requested.");
      }
      return { client: openRouterClient, type: 'openrouter', model: agentModel };
  } else if (agentModel?.startsWith('command')) { // Assuming Cohere models start with 'command'
      if (!process.env.CO_API_KEY) {
         throw new Error("CO_API_KEY is not configured, but a Cohere model was requested.");
      }
      return { client: cohereClient, type: 'cohere', model: agentModel };
  } else {
      // Default or throw error
      logger.warn(`Unknown model prefix: ${agentModel}. Defaulting to Cohere.`);
       if (!process.env.CO_API_KEY) {
         throw new Error("CO_API_KEY is not configured for default backend.");
      }
      return { client: cohereClient, type: 'cohere', model: agentModel || 'command-light' }; // Default model
  }
}

export class LLMService {
  static #currentBackend = 'cohere'; // Private static field for the current backend
  static #cohereClient = null;
  static #openRouterClient = null;

  // --- ADD OR VERIFY THIS STATIC FUNCTION ---
  static setBackend(backendName) {
    console.log(`Attempting to set backend to: ${backendName}`); // Add for debugging
    if (backendName === 'cohere') {
      if (!process.env.CO_API_KEY) {
        throw new Error("CO_API_KEY is not configured in .env for the Cohere backend.");
      }
      if (!LLMService.#cohereClient) {
        LLMService.#cohereClient = new CohereClient({ token: process.env.CO_API_KEY });
      }
      LLMService.#currentBackend = 'cohere';
      console.log("Backend set to Cohere");
    } else if (backendName === 'openrouter') {
      if (!process.env.OPENROUTER_API_KEY) {
        throw new Error("OPENROUTER_API_KEY is not configured in .env for the OpenRouter backend.");
      }
      if (!LLMService.#openRouterClient) {
        LLMService.#openRouterClient = new OpenAI({
          baseURL: "https://openrouter.ai/api/v1",
          apiKey: process.env.OPENROUTER_API_KEY,
        });
      }
      LLMService.#currentBackend = 'openrouter';
      console.log("Backend set to OpenRouter");
    } else {
      throw new Error(`Unsupported backend: ${backendName}. Use 'cohere' or 'openrouter'.`);
    }
  }

  // --- Helper Function to Get Client/Backend ---
  // Make sure this uses the static field #currentBackend
  static getBackendConfig(agentModel) {
    // Check LLMService.#currentBackend first, potentially simplifying logic
    if (LLMService.#currentBackend === 'openrouter') {
      if (!LLMService.#openRouterClient) throw new Error("OpenRouter client requested but not initialized. Call setBackend('openrouter') first.");
      // Determine the specific OpenRouter model (could be based on agentModel or a default)
      const model = agentModel || 'openai/gpt-4o-mini'; // Example: Use agent model or default
      console.log(`Using OpenRouter backend with model: ${model}`);
      return { client: LLMService.#openRouterClient, type: 'openrouter', model: model };
    } else { // Default to cohere
      if (!LLMService.#cohereClient) throw new Error("Cohere client requested but not initialized. Call setBackend('cohere') first.");
      // Determine the specific Cohere model
      const model = agentModel || 'command-r'; // Example: Use agent model or default
      console.log(`Using Cohere backend with model: ${model}`);
      return { client: LLMService.#cohereClient, type: 'cohere', model: model };
    }

    // --- Original Prefix-based logic (Can be kept as fallback or removed if #currentBackend is reliable) ---
    /*
    if (agentModel?.startsWith('openai/') || agentModel?.startsWith('google/') || agentModel?.startsWith('mistralai/')) {
      if (!LLMService.#openRouterClient) throw new Error("OPENROUTER_API_KEY not set or client not initialized.");
      return { client: LLMService.#openRouterClient, type: 'openrouter', model: agentModel };
    } else { // Default to Cohere
      if (!LLMService.#cohereClient) throw new Error("CO_API_KEY not set or client not initialized.");
      return { client: LLMService.#cohereClient, type: 'cohere', model: agentModel || 'command-r' };
    }
    */
  }

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
    let backendConfig; // Declare here to access in catch block

    try {
      // Get the dynamic backend configuration
      backendConfig = LLMService.getBackendConfig(agent.model);

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
        loggingPayload.input = { prompt: prompt, gameState: gameState }; // Added gameState for context if needed later

        let responseText = '';
        let requestId = null; // Only relevant for ART (potentially OpenRouter later?)
        let llmResponse;

        // Use backendConfig.type to determine the call
        if (backendConfig.type === 'openrouter') {
            logger.debug(`[OPENROUTER] Calling API for ${taskType}...`);
            if (!backendConfig.client) throw new Error('OpenRouter client not initialized via setBackend.');

            const completion = await backendConfig.client.chat.completions.create({
                 extra_headers: {
                   "HTTP-Referer": process.env.YOUR_SITE_URL || "http://localhost:3000",
                   "X-Title": process.env.YOUR_SITE_NAME || "Cluedo Arena",
                 },
                 model: backendConfig.model,
                 messages: [ { role: "user", content: prompt } ],
                 // Consider adding temperature: 0.1 or similar
                 // response_format: { type: "json_object" } // If the OpenRouter model supports it
            });
            responseText = completion.choices[0]?.message?.content;
             if (!responseText) throw new Error("Empty response content from OpenRouter.");
            loggingPayload.output = responseText; // Log raw OpenRouter output

        } else if (backendConfig.type === 'cohere') {
            logger.debug(`[COHERE] Calling API for ${taskType}...`);
            if (!backendConfig.client) throw new Error('Cohere client not initialized via setBackend.');

            const supportsJsonResponseFormat = backendConfig.model.startsWith('command-r');
            const apiParams = { model: backendConfig.model, message: prompt, temperature: 0.1 };
            if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };

            llmResponse = await backendConfig.client.chat(apiParams);
            responseText = llmResponse.text;
            loggingPayload.output = responseText; // Log raw Cohere output
        } else {
             // If ART or other backends were supported, handle them here
             // For now, assume only cohere and openrouter are configured via setBackend
             throw new Error(`Unsupported backend type configured: ${backendConfig.type}`);
        }


        // --- Parsing and Validation (Common Logic) ---
        let parsedResult = extractJSON(responseText);
        if (!parsedResult || typeof parsedResult !== 'object') {
            loggingPayload.validationStatus = 'failed_parsing';
            throw new Error(`Failed to parse JSON response from LLM (${backendConfig.type})`);
        }
        loggingPayload.parsedOutput = parsedResult;

        if (!parsedResult.suspect || !parsedResult.weapon || !parsedResult.room || !parsedResult.reasoning) {
          loggingPayload.validationStatus = 'failed_validation';
            throw new Error(`LLM response (${backendConfig.type}) missing required fields`);
      }
      if (parsedResult.room !== agent.location) {
            logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} suggestion: Room mismatch. Agent in ${agent.location}, suggested ${parsedResult.room}. Overriding.`);
           parsedResult.room = agent.location;
           loggingPayload.validationStatus = 'corrected_room';
      } else {
           loggingPayload.validationStatus = 'passed';
      }
        if (!gameState.availableSuspects.includes(parsedResult.suspect) || !gameState.availableWeapons.includes(parsedResult.weapon)) {
            logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} suggestion: Invalid suspect or weapon suggested. S:${parsedResult.suspect}, W:${parsedResult.weapon}`);
            // TODO: Decide how to handle - fallback or error? For now, allow but warn.
        }

        // --- Logging (Always log successful or corrected interactions) ---
        await LoggingService.logLLMInteraction(loggingPayload);

        logger.info(`[${backendConfig.type.toUpperCase()}] Suggestion for ${agent.name} took ${Date.now() - startTime}ms`);

        // Return result
        const finalResult = {
            suspect: parsedResult.suspect,
            weapon: parsedResult.weapon,
            room: parsedResult.room,
            reasoning: parsedResult.reasoning
        };
        // Add requestId if applicable (e.g., for ART)
        // if (requestId) finalResult.requestId = requestId;
        return finalResult;

    } catch (error) {
        // Use backendConfig.type in error messages if backendConfig was successfully retrieved
        const backendType = backendConfig ? backendConfig.type.toUpperCase() : 'UNKNOWN_BACKEND';
        logger.error(`[${backendType}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
        loggingPayload.error = error.message;
        loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

        // --- Logging (Log failed interactions) ---
        // Ensure logging happens even on error
        try {
             await LoggingService.logLLMInteraction(loggingPayload);
        } catch (logError) {
             logger.error(`Failed to log LLM interaction error: ${logError.message}`);
        }

        // Consistent fallback structure
        return {
            suspect: gameState.availableSuspects ? gameState.availableSuspects[0] : 'Miss Scarlet',
            weapon: gameState.availableWeapons ? gameState.availableWeapons[0] : 'Candlestick',
            room: agent.location || 'Lounge',
            reasoning: `Error occurred during ${taskType} via ${backendType}: ${error.message}, using fallback.`,
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
      const loggingPayload = { // Common structure for logging
          type: taskType,
          agent: agent.name,
          model: agent.model,
          input: {},
          output: null,
          error: null,
          parsedOutput: null,
          validationStatus: 'pending'
      };
      let backendConfig; // Declare here for access in catch block

      if (!turnEvents || turnEvents.length === 0) {
          // Consistent return structure for skipped update
          return { deducedCards: [], summary: '(Memory update skipped, no events)' };
      }

      try {
          // Get the dynamic backend configuration
          backendConfig = LLMService.getBackendConfig(agent.model);
          const formattedMemory = await memory.formatMemoryForLLM();
          const prompt = `You are ${agent.name}. Analyze the events from your last turn and update your memory and deductions.\\n\\nWHAT IS A DEDUCTION:\\nA deduction is a card that you can definitively conclude is NOT part of the murder solution. You can deduce a card when:\\n1. It's in your hand (you can see it, so it can't be part of the solution)\\n2. Another player shows it to you (proving it's not in the solution)\\n3. You can logically prove it must be held by a specific player based on the game events\\n\\nYour current knowledge:\\nCards in my hand: ${Array.from(agent.cards).join(', ')}\\nKnown Information: ${JSON.stringify(formattedMemory.knownInformation, null, 2)}\\nYour most recent memory note:\\n${formattedMemory.currentDeductions}\\n\\nEvents from my last turn:\\n${turnEvents.map(event => event.replace(agent.name, 'I').replace(/^I showed/, 'I showed').replace(/showed you/, 'showed me')).join('\n')}\\n\\nBased ONLY on the information above, what new cards can you definitively deduce are NOT part of the solution?\\nRemember: A deduction must be 100% certain - do not include guesses or probabilities.\\n\\nRespond ONLY with a JSON object in the following format. Provide an empty list if no new cards were deduced.\\nIMPORTANT: Do NOT use markdown code blocks (\`\`\`json) in your response - just return the JSON object directly.\\n\\n{\\n  \"newlyDeducedCards\": [\"string\"],\\n  \"reasoning\": \"string (explain exactly how you know each newly deduced card cannot be part of the solution)\",\\n  \"memorySummary\": \"string (Provide a DETAILED summary of your CURRENT understanding of the game state. Include ALL cards you know are eliminated (your hand + deduced), any strong suspicions, and key insights derived from the entire game history, not just the last turn.)\"\\n}`;
          loggingPayload.input = { prompt, turnEvents }; // Include turnEvents in log input

          let responseText = '';
          let requestId = null; // If needed for OpenRouter/ART in future
          let llmResponse;

          if (backendConfig.type === 'openrouter') {
              logger.debug(`[OPENROUTER] Calling API for ${taskType}...`);
              if (!backendConfig.client) throw new Error('OpenRouter client not initialized via setBackend.');

              const completion = await backendConfig.client.chat.completions.create({
                 extra_headers: {
                   "HTTP-Referer": process.env.YOUR_SITE_URL || "http://localhost:3000",
                   "X-Title": process.env.YOUR_SITE_NAME || "Cluedo Arena",
                 },
                 model: backendConfig.model,
                 messages: [ { role: "user", content: prompt } ],
                 // response_format: { type: "json_object" } // If model supports
              });
              responseText = completion.choices[0]?.message?.content;
              if (!responseText) throw new Error("Empty response content from OpenRouter.");
              loggingPayload.output = responseText;

          } else if (backendConfig.type === 'cohere') {
              logger.debug(`[COHERE] Calling API for ${taskType}...`);
              if (!backendConfig.client) throw new Error('Cohere client not initialized via setBackend.');

              const supportsJsonResponseFormat = backendConfig.model.startsWith('command-r');
              const apiParams = { model: backendConfig.model, message: prompt, temperature: 0.1 };
              if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };

              llmResponse = await backendConfig.client.chat(apiParams);
              responseText = llmResponse.text;
              loggingPayload.output = responseText;

          } else {
             throw new Error(`Unsupported backend type configured: ${backendConfig.type}`);
          }

          // --- Parsing and Validation ---
          const parsedResult = extractJSON(responseText);
          if (!parsedResult || typeof parsedResult !== 'object') {
              loggingPayload.validationStatus = 'failed_parsing';
              throw new Error(`Failed to parse JSON response from LLM (${backendConfig.type}) for ${taskType}`);
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

          // --- Update Memory Object ---
          if (memory.update) {
              await memory.update(summary, deducedCards, reasoning);
          } else {
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name}: Memory object does not have an update method.`);
          }

          // --- Logging ---
          await LoggingService.logLLMInteraction(loggingPayload);

          logger.info(`[${backendConfig.type.toUpperCase()}] ${taskType} for ${agent.name} took ${Date.now() - startTime}ms`);

          // --- Return Result ---
          const finalResult = { deducedCards, summary };
          // if (requestId) finalResult.requestId = requestId;
          return finalResult;

      } catch (error) {
          const backendType = backendConfig ? backendConfig.type.toUpperCase() : 'UNKNOWN_BACKEND';
          logger.error(`[${backendType}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
          loggingPayload.error = error.message;
          loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

          // Log error
          try {
             await LoggingService.logLLMInteraction(loggingPayload);
          } catch (logError) {
             logger.error(`Failed to log LLM interaction error: ${logError.message}`);
          }

          // Consistent fallback structure
          return {
            deducedCards: [],
            summary: `(Error during ${taskType} via ${backendType}: ${error.message})`,
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
      const loggingPayload = { // Common structure
          type: taskType,
          agent: agent.name,
          model: agent.model,
          input: {},
          output: null,
          error: null,
          parsedOutput: null,
          validationStatus: 'pending'
      };
      let backendConfig;

      if (!cards || cards.length === 0) {
          // No need to call LLM if no cards match
          return { cardToShow: null, reasoning: "No matching cards to show" };
      }

      try {
          backendConfig = LLMService.getBackendConfig(agent.model);
          const memoryState = await agent.memory.formatMemoryForLLM();
          const prompt = `You received a suggestion: ${suggestion.suspect}, ${suggestion.weapon}, ${suggestion.room}.\nYou hold the following matching card(s): ${cards.join(', ')}.\n\nYour current knowledge:\nKnown cards held: ${Array.from(agent.cards).join(', ') || 'None'}\nKnown Information: ${JSON.stringify(memoryState.knownInformation, null, 2)}\nCurrent Deductions: ${memoryState.currentDeductions}\n\nChoose ONE card from your matching cards (${cards.join(', ')}) to show to the suggester. Consider which card reveals the least about your overall hand and deductions, while still disproving the suggestion.\n\nRespond ONLY with a JSON object in the following format.\nIMPORTANT: Do NOT use markdown code blocks (\`\`\`json) in your response - just return the JSON object directly.\n\n{\n  "cardToShow": "string (must be one of: ${cards.join(', ')})",\n  "reasoning": "string (briefly explain your choice)"\n}`;
          loggingPayload.input = { prompt, suggestion, cards }; // Log relevant inputs

          let responseText = '';
          let requestId = null;
          let llmResponse;

          if (backendConfig.type === 'openrouter') {
              logger.debug(`[OPENROUTER] Calling API for ${taskType}...`);
              if (!backendConfig.client) throw new Error('OpenRouter client not initialized via setBackend.');

              const completion = await backendConfig.client.chat.completions.create({
                 extra_headers: {
                   "HTTP-Referer": process.env.YOUR_SITE_URL || "http://localhost:3000",
                   "X-Title": process.env.YOUR_SITE_NAME || "Cluedo Arena",
                 },
                 model: backendConfig.model,
                 messages: [ { role: "user", content: prompt } ],
                 // response_format: { type: "json_object" } // If model supports
              });
              responseText = completion.choices[0]?.message?.content;
              if (!responseText) throw new Error("Empty response content from OpenRouter.");
              loggingPayload.output = responseText;

          } else if (backendConfig.type === 'cohere') {
              logger.debug(`[COHERE] Calling API for ${taskType}...`);
              if (!backendConfig.client) throw new Error('Cohere client not initialized via setBackend.');

              const supportsJsonResponseFormat = backendConfig.model.startsWith('command-r');
              const apiParams = { model: backendConfig.model, message: prompt, temperature: 0.1 };
              if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };

              llmResponse = await backendConfig.client.chat(apiParams);
              responseText = llmResponse.text;
              loggingPayload.output = responseText;

          } else {
             throw new Error(`Unsupported backend type configured: ${backendConfig.type}`);
          }

          // --- Parsing and Validation ---
          let parsedResult = extractJSON(responseText);
           if (!parsedResult || typeof parsedResult !== 'object' || !parsedResult.cardToShow) {
              loggingPayload.validationStatus = 'failed_parsing';
              // Don't throw an error here, fallback logic below handles it
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: Failed to parse JSON response or missing cardToShow. Response: ${responseText}`);
              parsedResult = { cardToShow: cards[0], reasoning: `(Fallback: Failed to parse LLM response)` }; // Ensure parsedResult is an object for fallback
          } else {
             loggingPayload.parsedOutput = parsedResult; // Log only if parsing succeeded initially
          }

          const cardToShow = parsedResult.cardToShow;
          let reasoning = parsedResult.reasoning || '(No reasoning provided by LLM)';

          if (!cards.includes(cardToShow)) {
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: LLM chose invalid card (${cardToShow}). Not in matching set (${cards.join(', ')}). Falling back.`);
              const fallbackCard = cards[0];
              reasoning = `(Fallback: LLM chose invalid card ${cardToShow}). ${reasoning}`; // Prepend fallback reason
              loggingPayload.validationStatus = 'corrected_invalid_card';
              // Log the original invalid response before returning fallback
              await LoggingService.logLLMInteraction(loggingPayload);
              return { cardToShow: fallbackCard, reasoning: reasoning };
          } else {
              loggingPayload.validationStatus = 'passed'; // Or failed_parsing if initial parse failed but we handled it
               if (loggingPayload.parsedOutput) loggingPayload.validationStatus = 'passed'; // Mark as passed only if initial parse was okay
          }

          // --- Logging (Successful or handled parse failure) ---
          await LoggingService.logLLMInteraction(loggingPayload);

          logger.info(`[${backendConfig.type.toUpperCase()}] ${taskType} for ${agent.name} took ${Date.now() - startTime}ms`);

          // --- Return Result ---
          const finalResult = { cardToShow, reasoning };
          // if (requestId) finalResult.requestId = requestId;
          return finalResult;

    } catch (error) {
          const backendType = backendConfig ? backendConfig.type.toUpperCase() : 'UNKNOWN_BACKEND';
          logger.error(`[${backendType}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
          loggingPayload.error = error.message;
          loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

          // Log error
          try {
             await LoggingService.logLLMInteraction(loggingPayload);
          } catch (logError) {
             logger.error(`Failed to log LLM interaction error: ${logError.message}`);
          }

          // Consistent fallback structure
          const fallbackCard = cards[0] || null; // Ensure fallback exists
          return {
              cardToShow: fallbackCard,
              reasoning: `(Fallback: Error during ${taskType} via ${backendType}: ${error.message}). Showing ${fallbackCard || 'nothing'}.`,
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
      const loggingPayload = { // Common structure
          type: taskType,
          agent: agent.name,
          model: agent.model,
          input: {},
          output: null,
          error: null,
          parsedOutput: null,
          validationStatus: 'pending'
      };
      let backendConfig;

    try {
      backendConfig = LLMService.getBackendConfig(agent.model);
      const memoryState = await agent.memory.formatMemoryForLLM();
      // Extract current turn info from gameState
      const currentSuggestion = gameState.currentSuggestion;
      const currentChallengeResult = gameState.currentChallengeResult;

      let currentTurnEventsString = "No suggestion made this turn.";
      if (currentSuggestion) {
          currentTurnEventsString = `This turn, you suggested: ${currentSuggestion.suspect}, ${currentSuggestion.weapon}, ${currentSuggestion.room}.\n`;
          if (currentChallengeResult && currentChallengeResult.cardToShow) {
              currentTurnEventsString += `Result: ${currentChallengeResult.challengingAgent} showed you the card: ${currentChallengeResult.cardToShow}.`;
          } else {
              currentTurnEventsString += `Result: NO ONE could challenge your suggestion.`; // Highlight this crucial outcome
          }
      }

          const prompt = `Based on your complete knowledge AND the events of THIS turn, decide if you are confident enough to make a final accusation to win the game.

Your knowledge:
- Your Hand: ${Array.from(agent.cards).join(', ') || 'None'}
- Structured Knowledge: ${JSON.stringify(memoryState.knownInformation, null, 2)}
- Your Most Recent Memory Note: ${memoryState.currentDeductions}

Previous Turn Summary:
${memoryState.turnHistory.join('\n')}

Current Turn Events (Turn ${gameState.currentTurn}):
${currentTurnEventsString}

IMPORTANT CLUEDO LOGIC:
1. If a suggestion is made and NO PLAYER can challenge it (show any cards), this is strong evidence that ALL THREE suggested cards might be in the solution.
2. If a suggestion is made and is challenged, at least ONE of the suggested cards is NOT in the solution.
3. Through elimination: If you can identify all but one card of a category (e.g., 5 of 6 suspects), the remaining one MUST be the solution.
4. You can win by making a correct accusation even without 100% certainty - reasonable deduction based on probabilities is valid.

You should consider making an accusation when:
- You have strong evidence for all three components (suspect, weapon, room)
- A suggestion including certain cards was not challenged by any player
- Through the process of elimination, you've narrowed down possibilities significantly
- The potential reward of winning outweighs the risk of being wrong

Respond ONLY with a JSON object in the following format.
If shouldAccuse is true, provide your deduced solution.
If shouldAccuse is false, provide null for accusation components.
IMPORTANT: Do NOT use markdown code blocks (\`\`\`json) in your response - just return the JSON object directly.

{
  "shouldAccuse": boolean,
  "accusation": {
    "suspect": "string | null (your deduced suspect or null)",
    "weapon": "string | null (your deduced weapon or null)",
    "room": "string | null (your deduced room or null)"
  },
  "reasoning": "string (explain your decision and reasoning)"
}`;
          loggingPayload.input = { prompt: prompt, gameState: gameState }; // Added gameState for context if needed later

          let responseText = '';
          let requestId = null;
          let llmResponse;

          if (backendConfig.type === 'openrouter') {
              logger.debug(`[OPENROUTER] Calling API for ${taskType}...`);
              if (!backendConfig.client) throw new Error('OpenRouter client not initialized via setBackend.');

              const completion = await backendConfig.client.chat.completions.create({
                 extra_headers: {
                   "HTTP-Referer": process.env.YOUR_SITE_URL || "http://localhost:3000",
                   "X-Title": process.env.YOUR_SITE_NAME || "Cluedo Arena",
                 },
                 model: backendConfig.model,
                 messages: [ { role: "user", content: prompt } ],
                 // response_format: { type: "json_object" } // If model supports
              });
              responseText = completion.choices[0]?.message?.content;
              if (!responseText) throw new Error("Empty response content from OpenRouter.");
              loggingPayload.output = responseText;

          } else if (backendConfig.type === 'cohere') {
              logger.debug(`[COHERE] Calling API for ${taskType}...`);
              if (!backendConfig.client) throw new Error('Cohere client not initialized via setBackend.');

              const supportsJsonResponseFormat = backendConfig.model.startsWith('command-r');
              const apiParams = { model: backendConfig.model, message: prompt, temperature: 0.1 };
              if (supportsJsonResponseFormat) apiParams.response_format = { type: "json_object" };

              llmResponse = await backendConfig.client.chat(apiParams);
              responseText = llmResponse.text;
              loggingPayload.output = responseText;

          } else {
             throw new Error(`Unsupported backend type configured: ${backendConfig.type}`);
          }

          // --- Parsing and Validation ---
          // Use safeParseJSON for accusation which includes validation schema and normalization
          const validationResult = safeParseJSON(responseText, accusationSchema);
          if (!validationResult.valid) {
              loggingPayload.validationStatus = 'failed_validation';
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: Failed validation: ${JSON.stringify(validationResult.error)}. Response: ${responseText}`);
              // Don't throw error, use fallback below
              // Set a default non-accusing structure for fallback logic
              validationResult.data = {
                  shouldAccuse: false,
                  accusation: { suspect: null, weapon: null, room: null },
                  reasoning: `(Fallback: LLM response failed validation: ${JSON.stringify(validationResult.error)})`
              };
          }
          const parsedResult = validationResult.data;
          loggingPayload.parsedOutput = parsedResult; // Log the (potentially corrected) data
          loggingPayload.validationStatus = validationResult.valid ? 'passed' : 'failed_validation'; // Reflect validation status

          // Additional logic check (even if JSON is valid)
          if (parsedResult.shouldAccuse &&
              (!parsedResult.accusation.suspect || !parsedResult.accusation.weapon || !parsedResult.accusation.room)) {
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: shouldAccuse is true but accusation details missing/null. Overriding to false.`);
              parsedResult.shouldAccuse = false;
              parsedResult.reasoning += " (Invalid accusation details provided, overriding shouldAccuse to false)";
              loggingPayload.validationStatus = 'corrected_logic'; // Mark as corrected
          }

          // --- Logging ---
          await LoggingService.logLLMInteraction(loggingPayload);

          logger.info(`[${backendConfig.type.toUpperCase()}] ${taskType} for ${agent.name} took ${Date.now() - startTime}ms`);

          // --- Return Result ---
          const finalResult = {
              shouldAccuse: parsedResult.shouldAccuse,
              accusation: parsedResult.accusation,
              reasoning: parsedResult.reasoning
          };
          // if (requestId) finalResult.requestId = requestId;
          return finalResult;

    } catch (error) {
        const backendType = backendConfig ? backendConfig.type.toUpperCase() : 'UNKNOWN_BACKEND';
        logger.error(`[${backendType}] ${agent.name} failed ${taskType}: ${error.message}`, { error });
        loggingPayload.error = error.message;
        loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;

          // Log error
          try {
             await LoggingService.logLLMInteraction(loggingPayload);
          } catch (logError) {
             logger.error(`Failed to log LLM interaction error: ${logError.message}`);
          }

          // Consistent fallback structure
          return {
            shouldAccuse: false,
            accusation: { suspect: null, weapon: null, room: null },
            reasoning: `(Fallback: Error during ${taskType} via ${backendType}: ${error.message}). Defaulting to not accuse.`,
            error: error.message || `Unknown error during ${taskType}`
          };
    }
  }
} 