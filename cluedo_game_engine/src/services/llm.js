import 'dotenv/config';
import Ajv from 'ajv';
import axios from 'axios';
import { SUSPECTS, WEAPONS, ROOMS } from '../config/gameConstants.js';
import { logger } from '../utils/logger.js';
import fs from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';
import { CohereClient } from 'cohere-ai';
import { LoggingService } from './LoggingService.js';
import { OpenAI } from 'openai';       // Import OpenAI SDK
import dotenv from 'dotenv';
import yaml from 'js-yaml'; // <-- Add YAML import

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
  'anthropic/claude-3.5-sonnet',          
  'anthropic/claude-3.5-sonnet',     
  'anthropic/claude-3.5-sonnet',      
  'anthropic/claude-3.5-sonnet',          
  'anthropic/claude-3.5-sonnet',     
  'anthropic/claude-3.5-sonnet'     
];

// Initialize JSON schema validator
const ajv = new Ajv();

// Define JSON schemas (Only accusationSchema is actively used for validation now)
const suggestionSchema = {
  type: "object",
  properties: {
    suspect: { type: "string", enum: SUSPECTS },
    weapon: { type: "string", enum: WEAPONS },
    room: { type: "string", enum: ROOMS },
    reasoning: { type: "string", minLength: 1 }
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
        suspect: { type: ["string", "null"], enum: [...SUSPECTS, null] },
        weapon: { type: ["string", "null"], enum: [...WEAPONS, null] },
        room: { type: ["string", "null"], enum: [...ROOMS, null] }
      },
      required: ["suspect", "weapon", "room"],
    },
    reasoning: { type: "string" }
  },
  required: ["shouldAccuse", "accusation", "reasoning"],
  additionalProperties: false
};

const memoryUpdateSchema = {
    type: 'object',
    properties: {
        newlyDeducedCards: { type: 'array', items: { type: 'string' } },
        reasoning: { type: 'string' },
        memorySummary: { type: 'string' }
    },
    required: ['newlyDeducedCards', 'reasoning', 'memorySummary'],
    additionalProperties: false
};

const challengeSchema = {
    type: 'object',
    properties: {
        cardToShow: { type: 'string' },
        reasoning: { type: 'string' }
    },
    required: ['cardToShow', 'reasoning'],
    additionalProperties: false
};

const validateSuggestion = ajv.compile(suggestionSchema);
const validateAccusation = ajv.compile(accusationSchema);
const validateMemoryUpdate = ajv.compile(memoryUpdateSchema);
const validateChallenge = ajv.compile(challengeSchema);

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
   * Attempts to parse a YAML string from the LLM response.
   * Handles potential errors during parsing.
   *
   * @param {string} response - The raw string response from the LLM.
   * @returns {object | null} The parsed JavaScript object or null if parsing fails.
   * @private
   */
  static #extractYAML(response) {
    if (!response || typeof response !== 'string') {
      return null;
    }
    try {
      // Check for markdown fences and extract content if present
      const yamlMatch = response.match(/```(?:yaml)?\n?([\s\S]*?)\n?```/);
      const yamlContent = yamlMatch ? yamlMatch[1] : response;
      
      const parsed = yaml.load(yamlContent.trim()); // Trim whitespace
      if (parsed !== null && typeof parsed === 'object') {
        return parsed;
      }
      logger.warn(`YAML parsing resulted in non-object type: ${typeof parsed}`);
      return null;
    } catch (e) {
      logger.error(`Failed to parse YAML: ${e.message}`, { response });
      return null;
    }
  }

  /**
   * Parses YAML response and validates it against a given Ajv schema.
   *
   * @param {string} response - The YAML string response from the LLM.
   * @param {Function} validateFunction - The compiled Ajv validation function.
   * @returns {{valid: boolean, data: object | null, error: string | null}} Validation result.
   * @private
   */
  static #safeParseYAML(response, validateFunction) {
    const parsed = LLMService.#extractYAML(response);
    if (!parsed) {
      return { valid: false, data: null, error: 'Failed to parse YAML response.' };
    }

    const isValid = validateFunction(parsed);
    if (!isValid) {
      return {
          valid: false,
          data: parsed, // Return parsed data even if invalid
          error: ajv.errorsText(validateFunction.errors)
      };
    }

    return { valid: true, data: parsed, error: null };
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
        const prompt = `You are the Cluedo agent ${agent.name}. Your turn ${gameState.currentTurn}.
Your hand: ${Array.from(agent.cards).join(', ')}.
Your current location: ${agent.location || 'Unknown (must be in a room to suggest)'}.
Available Rooms: ${gameState.availableRooms.join(', ')}

Your knowledge:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
Eliminated Cards (Not in Solution): ${Array.from(agent.memory.eliminatedCards).join(', ') || 'None'}
Suspected Cards: ${JSON.stringify(Object.fromEntries(agent.memory.suspectedCards))}
Current Deductions Summary: ${memoryState.currentDeductions}
Turn History Highlights:
${memoryState.turnHistory.join('\n')}

Based on your knowledge and location (${agent.location}), make a strategic suggestion (suspect, weapon, room).
The suggested room MUST be your current location: ${agent.location}.
Your goal is to gain new information by forcing others to reveal cards. Choose a suggestion that includes cards you suspect might be the solution OR cards held by others. Avoid suggesting only cards you know are eliminated unless tactically necessary.

Respond ONLY with a YAML object in the following format. Provide concise reasoning.

suspect: <string, one of ${gameState.availableSuspects.join(' | ')}>
weapon: <string, one of ${gameState.availableWeapons.join(' | ')}>
room: <string, MUST be ${agent.location}>
reasoning: <string, your detailed thought process for this suggestion>`;
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


        // --- Parsing and Validation (YAML) ---
        const validationResult = LLMService.#safeParseYAML(responseText, validateSuggestion);
        loggingPayload.parsedOutput = validationResult.data; // Log parsed data regardless of validity

        if (!validationResult.valid) {
          loggingPayload.validationStatus = 'failed_validation';
          logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: Failed YAML validation: ${validationResult.error}. Response: ${responseText}`);
          // Fallback logic
          const fallbackSuspect = gameState.availableSuspects.find(s => !agent.memory.eliminatedCards.has(s)) || gameState.availableSuspects[0];
          const fallbackWeapon = gameState.availableWeapons.find(w => !agent.memory.eliminatedCards.has(w)) || gameState.availableWeapons[0];
          const fallbackReasoning = `(Fallback: LLM response failed YAML validation: ${validationResult.error})`;

          // Log and return fallback
          await LoggingService.logLLMInteraction(loggingPayload);
          return {
            suspect: fallbackSuspect,
            weapon: fallbackWeapon,
            room: agent.location, // Must use agent's location
            reasoning: fallbackReasoning,
            error: `Failed validation: ${validationResult.error}`
          };
        }

        // Additional Logic Check (e.g., ensure room matches location)
        const parsedSuggestion = validationResult.data;
        if (parsedSuggestion.room !== agent.location) {
             logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: LLM suggested room (${parsedSuggestion.room}) different from agent location (${agent.location}). Correcting.`);
             parsedSuggestion.room = agent.location; // Force correct room
             parsedSuggestion.reasoning += ` (Corrected room to agent's location: ${agent.location})`;
             loggingPayload.validationStatus = 'corrected_room';
             loggingPayload.parsedOutput = parsedSuggestion; // Log corrected data
        } else {
             loggingPayload.validationStatus = 'passed';
        }
        // --- End Parsing and Validation ---

        // --- Logging (Always log successful or corrected interactions) ---
        await LoggingService.logLLMInteraction(loggingPayload);

        logger.info(`[${backendConfig.type.toUpperCase()}] Suggestion for ${agent.name} took ${Date.now() - startTime}ms`);

        // Return result
        const finalResult = {
            suspect: parsedSuggestion.suspect,
            weapon: parsedSuggestion.weapon,
            room: parsedSuggestion.room,
            reasoning: parsedSuggestion.reasoning
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
        const fallbackSuspect = gameState.availableSuspects ? gameState.availableSuspects[0] : 'Miss Scarlet';
        const fallbackWeapon = gameState.availableWeapons ? gameState.availableWeapons[0] : 'Candlestick';
        return {
            suspect: fallbackSuspect,
            weapon: fallbackWeapon,
            room: agent.location || 'Lounge',
            reasoning: `(Fallback: Error during ${taskType} via ${backendType}: ${error.message}, using fallback.`,
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
          const prompt = `You are ${agent.name}. Analyze the events from your last turn (Turn ${agent.game.currentTurn}) and update your memory and deductions.

WHAT IS A DEDUCTION:
A deduction is a card that you can definitively conclude is NOT part of the murder solution. Deduce cards when:
1. It's in your hand.
2. Another player shows it to you.
3. You can logically prove it must be held by a specific player or eliminated.

Your current knowledge:
Cards in my hand: ${Array.from(agent.cards).join(', ')}
Known Eliminated Cards: ${Array.from(memory.eliminatedCards).join(', ') || 'None'}
Your most recent memory note:
${formattedMemory.currentDeductions}

Events from THIS turn:
${turnEvents.map(event => event.replace(agent.name, 'I').replace(/^I showed/, 'I showed').replace(/showed you/, 'showed me')).join('\n')}

Based ONLY on the information above, what new cards can you definitively deduce are NOT part of the solution?
Remember: A deduction must be 100% certain.

Respond ONLY with a YAML object in the following format. Provide a DETAILED summary.

newlyDeducedCards:
  - <string> # Card name, or empty list if none
reasoning: <string> # Explain exactly how you deduced each new card
memorySummary: <string> # DETAILED summary of your CURRENT understanding. Include ALL known eliminated cards (hand + deduced), suspicions, and key insights from the game history.`;
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

          // --- Parsing and Validation (YAML) ---
          const validationResult = LLMService.#safeParseYAML(responseText, validateMemoryUpdate);
          loggingPayload.parsedOutput = validationResult.data;

          if (!validationResult.valid) {
            loggingPayload.validationStatus = 'failed_validation';
            logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: Failed YAML validation: ${validationResult.error}. Response: ${responseText}`);
            // Provide fallback, ensuring structure matches expected return
            const fallbackResult = {
                deducedCards: [],
                summary: `(Fallback: LLM response failed YAML validation: ${validationResult.error})`,
                error: `Failed validation: ${validationResult.error}`
            };
            // Log error and return fallback
            await LoggingService.logLLMInteraction(loggingPayload);
            return fallbackResult;
          }

          loggingPayload.validationStatus = 'passed';
          const parsedResult = validationResult.data;
          const deducedCards = parsedResult.newlyDeducedCards || []; // Default to empty array
          const summary = parsedResult.memorySummary || parsedResult.reasoning || '(No summary provided by LLM)';
          const reasoning = parsedResult.reasoning || '(No reasoning provided by LLM)';

          // Ensure newlyDeducedCards is an array of strings (basic check)
          if (!Array.isArray(deducedCards) || !deducedCards.every(c => typeof c === 'string')) {
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: newlyDeducedCards is not an array of strings in YAML response. Correcting.`);
              loggingPayload.validationStatus = 'corrected_deductions_format';
              // Attempt to filter or handle, or just default to empty
              parsedResult.newlyDeducedCards = []; // Safest fallback
              loggingPayload.parsedOutput = parsedResult; // Log corrected data
          }
          // --- End Parsing and Validation ---

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
            summary: `(Fallback: Error during ${taskType} via ${backendType}: ${error.message})`,
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
          const prompt = `You are ${agent.name}. You received a suggestion: ${suggestion.suspect}, ${suggestion.weapon}, ${suggestion.room}.
You hold the following matching card(s): ${cards.join(', ')}.

Your current knowledge:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
Eliminated Cards (Not in Solution): ${Array.from(agent.memory.eliminatedCards).join(', ') || 'None'}
Current Deductions Summary: ${memoryState.currentDeductions}

Choose ONE card from your matching cards (${cards.join(', ')}) to show to the suggester.
Consider which card reveals the least about your overall hand and deductions.

Respond ONLY with a YAML object in the following format.

cardToShow: <string, must be one of: ${cards.join(' | ')}>
reasoning: <string, briefly explain your choice>`;
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

          // --- Parsing and Validation (YAML) ---
          const validationResult = LLMService.#safeParseYAML(responseText, validateChallenge);
          loggingPayload.parsedOutput = validationResult.data;

          if (!validationResult.valid || !validationResult.data?.cardToShow) { // Also check if cardToShow exists
              loggingPayload.validationStatus = 'failed_validation';
              const errorReason = validationResult.error || 'Missing cardToShow';
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: Failed YAML validation or missing cardToShow: ${errorReason}. Response: ${responseText}`);
              // Fallback logic
              const fallbackCard = cards[0]; // Show the first matching card
              const fallbackReasoning = `(Fallback: LLM response failed YAML validation: ${errorReason})`;

              await LoggingService.logLLMInteraction(loggingPayload);
              return { cardToShow: fallbackCard, reasoning: fallbackReasoning, error: `Failed validation: ${errorReason}` };
          }

          const parsedResult = validationResult.data;
          const cardToShow = parsedResult.cardToShow;
          let reasoning = parsedResult.reasoning || '(No reasoning provided by LLM)';

          // Check if the chosen card is valid
          if (!cards.includes(cardToShow)) {
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: LLM chose invalid card (${cardToShow}). Not in matching set (${cards.join(', ')}). Falling back.`);
              const fallbackCard = cards[0];
              reasoning = `(Fallback: LLM chose invalid card ${cardToShow}). ${reasoning}`;
              loggingPayload.validationStatus = 'corrected_invalid_card';
              loggingPayload.parsedOutput = { cardToShow: fallbackCard, reasoning }; // Log fallback data
              await LoggingService.logLLMInteraction(loggingPayload);
              return { cardToShow: fallbackCard, reasoning: reasoning };
          } else {
              loggingPayload.validationStatus = 'passed';
          }

          // --- Logging (Successful or handled validation failure) ---
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
          const fallbackCard = cards[0] || null;
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

          const prompt = `You are ${agent.name}. Turn ${gameState.currentTurn}. Decide if you should make a final accusation to win.

Your knowledge:
- Your Hand: ${Array.from(agent.cards).join(', ') || 'None'}
- Eliminated Cards (Not in Solution): ${Array.from(agent.memory.eliminatedCards).join(', ') || 'None'}
- Suspected Cards: ${JSON.stringify(Object.fromEntries(agent.memory.suspectedCards))}
- Current Deductions Summary: ${memoryState.currentDeductions}
- Turn History Highlights:
${memoryState.turnHistory.join('\n')}

Current Turn Events (Turn ${gameState.currentTurn}):
${currentTurnEventsString}

CLUEDO LOGIC REMINDERS:
1. No challenge to a suggestion implies suggested cards MIGHT be the solution.
2. A challenge proves AT LEAST ONE suggested card is NOT the solution.
3. Elimination: If 5/6 suspects are known, the last one IS the solution suspect.
4. You can risk an accusation without 100% certainty based on strong evidence.

Consider accusing if:
- You have strong evidence/elimination for all 3 solution components.
- A key suggestion was unchallenged.

Respond ONLY with a YAML object in the following format.
If shouldAccuse is true, provide your deduced solution.
If shouldAccuse is false, provide null for accusation components.

shouldAccuse: <boolean>
accusation:
  suspect: <string | null, one of ${SUSPECTS.join(' | ')} or null>
  weapon: <string | null, one of ${WEAPONS.join(' | ')} or null>
  room: <string | null, one of ${ROOMS.join(' | ')} or null>
reasoning: <string, explain your decision and confidence level>`;
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

          // --- Parsing and Validation (YAML) ---
          const validationResult = LLMService.#safeParseYAML(responseText, validateAccusation);
          loggingPayload.parsedOutput = validationResult.data; // Log potentially invalid data

          if (!validationResult.valid) {
              loggingPayload.validationStatus = 'failed_validation';
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: Failed YAML validation: ${validationResult.error}. Response: ${responseText}`);
              // Fallback: Don't accuse
              const fallbackResult = {
                  shouldAccuse: false,
                  accusation: { suspect: null, weapon: null, room: null },
                  reasoning: `(Fallback: LLM response failed YAML validation: ${validationResult.error})`,
                  error: `Failed validation: ${validationResult.error}`
              };
              await LoggingService.logLLMInteraction(loggingPayload);
              return fallbackResult; // Return structured fallback
          }

          const parsedResult = validationResult.data;
          loggingPayload.validationStatus = 'passed'; // Initial validation passed

          // Additional logic check (ensure details present if accusing)
          if (parsedResult.shouldAccuse &&
              (!parsedResult.accusation || !parsedResult.accusation.suspect || !parsedResult.accusation.weapon || !parsedResult.accusation.room)) {
              logger.warn(`[${backendConfig.type.toUpperCase()}] ${agent.name} ${taskType}: shouldAccuse is true but accusation details missing/null in YAML. Overriding to false.`);
              parsedResult.shouldAccuse = false;
              // Ensure accusation object exists before trying to null its properties
              if (!parsedResult.accusation) parsedResult.accusation = {};
              parsedResult.accusation.suspect = null;
              parsedResult.accusation.weapon = null;
              parsedResult.accusation.room = null;
              parsedResult.reasoning += " (Invalid accusation details provided, overriding shouldAccuse to false)";
              loggingPayload.validationStatus = 'corrected_logic'; // Mark as corrected
              loggingPayload.parsedOutput = parsedResult; // Log corrected data
          }

          // --- Logging ---
          await LoggingService.logLLMInteraction(loggingPayload);

          logger.info(`[${backendConfig.type.toUpperCase()}] ${taskType} for ${agent.name} took ${Date.now() - startTime}ms`);

          // --- Return Result ---
          const finalResult = {
              shouldAccuse: parsedResult.shouldAccuse,
              accusation: parsedResult.accusation, // Contains nulls if not accusing
              reasoning: parsedResult.reasoning
              // Include confidence if the schema/LLM provides it
              // confidence: parsedResult.confidence
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