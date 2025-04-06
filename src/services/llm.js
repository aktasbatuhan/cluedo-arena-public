import { CohereClient } from "cohere-ai";
import 'dotenv/config';
import Ajv from 'ajv';
import { Game } from '../models/Game.js';
import { logger } from '../utils/logger.js';
import { LoggingService } from './LoggingService.js';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// Add these near the top of the file with other imports
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

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
  'command-r7b-12-2024',     
  'command-r-plus-04-2024',      
  'command-r-08-2024',          
  'command-r-plus',     
  'c4ai-aya-expanse-32b'       
];

// Initialize Cohere client
const cohere = new CohereClient({
  token: process.env.COHERE_API_KEY,
  clientOptions: {
    timeoutInSeconds: 60 // Add a 60-second timeout
  }
});

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
    // Handle JSON wrapped in markdown code blocks
    const jsonMatch = response.match(/```(?:json)?\n([\s\S]*?)\n```/);
    if (jsonMatch) return JSON.parse(jsonMatch[1]);
    
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

/**
 * Service for interacting with Language Learning Models via OpenRouter.
 * 
 * This service provides methods for LLM-powered game actions:
 * - Making suggestions
 * - Evaluating challenges
 * - Considering accusations
 * - Updating agent memory
 * 
 * Each method handles model-specific prompting, response validation,
 * and error handling to ensure robust AI agent behavior.
 */
export class LLMService {
  /**
   * Generates a strategic suggestion for an agent during their turn.
   * 
   * @param {Agent} agent - The agent making the suggestion
   * @param {Object} gameState - Current game state information
   * @returns {Promise<Object>} Suggestion object with suspect, weapon, room, and reasoning
   */
  static async makeSuggestion(agent, gameState) {
    console.time(`[LLM] ${agent.name} suggestion`);
    const loggingPayload = {
      type: 'suggestion',
      agent: agent.name,
      model: agent.model,
      input: {},
      output: null,
      error: null,
      parsedOutput: null,
      validationStatus: 'pending'
    };

    try {
      const memoryState = await agent.memory.formatMemoryForLLM();
      
      const userMessage = `Analyze the game state and make a strategic suggestion:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
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
1. Your known cards and deductions (don't suggest cards you hold).
2. Previous suggestions and their outcomes (from Turn History).
3. Information revealed by others.
4. Your current room (${agent.location}) - you MUST suggest this room.
5. Choose a suspect (not yourself) and a weapon that seem most likely based on your deductions, or that would gather the most information.

Respond ONLY with a JSON object in the following format:
{
  "suspect": "string (must be an available suspect, not yourself)",
  "weapon": "string (must be an available weapon)",
  "room": "string (must be your current room: ${agent.location})",
  "reasoning": "string (explain your strategy and deduction process briefly)"
}`;

      loggingPayload.input = {
        prompt: userMessage,
        gameState: {
          knownCards: Array.from(agent.cards),
          currentTurn: gameState.currentTurn,
          location: agent.location,
          availableSuspects: gameState.availableSuspects.filter(s => s !== agent.name),
          availableWeapons: gameState.availableWeapons,
          memoryState
        }
      };
      
      logger.info(`[LLM Debug] Attempting cohere.chat call for ${agent.name} (Model: ${agent.model}) suggestion...`);
      console.log(`[LLM Debug] Attempting cohere.chat call for ${agent.name} (Model: ${agent.model}) suggestion...`);

      // Check if the model supports response_format (Command R and newer)
      const supportsJsonResponseFormat = agent.model.startsWith('command-r'); // Simple check for Command R models

      const apiParams = {
        model: agent.model,
        message: userMessage,
        temperature: 0.3,
      };

      if (supportsJsonResponseFormat) {
        apiParams.response_format = { type: "json_object" };
        logger.info(`[LLM Debug] Using response_format: json_object for model ${agent.model}`);
      } else {
        logger.info(`[LLM Debug] Model ${agent.model} does not support response_format: json_object. Relying on prompt.`);
      }

      const completion = await cohere.chat(apiParams);
      
      logger.info(`[LLM Debug] cohere.chat call for ${agent.name} (Model: ${agent.model}) suggestion completed.`);
      console.log(`[LLM Debug] cohere.chat call for ${agent.name} (Model: ${agent.model}) suggestion completed.`);

      const responseText = completion.text;
      loggingPayload.output = responseText;

      let parsedResult;
      try {
        // If response_format wasn't used, the text might contain markdown or just plain JSON
        // Use extractJSON which handles both cases
        if (!supportsJsonResponseFormat) {
          parsedResult = extractJSON(responseText);
        } else {
          // If response_format was used, the API should return clean JSON text
          parsedResult = JSON.parse(responseText);
        }

        if (!parsedResult) { // Check if parsing/extraction failed
            throw new Error('Failed to extract or parse JSON from response.');
        }
        loggingPayload.parsedOutput = parsedResult;

      } catch (parseError) {
        logger.error('Failed to parse suggestion JSON response from Cohere:', { error: parseError.message, responseText, model: agent.model });
        loggingPayload.error = `JSON Parsing Error: ${parseError.message}`;
        loggingPayload.validationStatus = 'failed_parsing';
        throw new Error('Failed to parse suggestion response as JSON.');
      }

      if (!parsedResult || typeof parsedResult !== 'object' || !parsedResult.suspect || !parsedResult.weapon || !parsedResult.room || !parsedResult.reasoning) {
          logger.error('Invalid structure in suggestion JSON response:', { parsedResult });
          loggingPayload.error = 'Invalid JSON structure received';
          loggingPayload.validationStatus = 'failed_validation';
          throw new Error('Invalid JSON structure in suggestion response.');
      }
      
      if (parsedResult.room !== agent.location) {
           logger.info(`LLM suggested wrong room (${parsedResult.room}) for agent ${agent.name} at ${agent.location}. Forcing correct room.`);
           parsedResult.room = agent.location;
           loggingPayload.validationStatus = 'corrected_room';
      } else {
           loggingPayload.validationStatus = 'passed';
      }

      await LoggingService.logLLMInteraction(loggingPayload);

      console.timeEnd(`[LLM] ${agent.name} suggestion`);
      return parsedResult;

    } catch (error) {
      logger.error(`Error in makeSuggestion for ${agent.name}: ${error.message}`, { stack: error.stack });
      console.timeEnd(`[LLM] ${agent.name} suggestion`);
      
      loggingPayload.error = error.message;
      loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;
      await LoggingService.logLLMInteraction(loggingPayload);

      throw new Error(`LLM suggestion failed for ${agent.name}: ${error.message}`);
    }
  }
  

  /**
   * Updates the agent's memory based on events from the completed turn.
   *
   * @param {Agent} agent - The agent whose memory to update
   * @param {AgentMemory} memory - The agent's current memory object
   * @param {Array<string>} turnEvents - List of events that occurred during the turn
   * @returns {Promise<AgentMemory>} The updated memory object
   */
  static async updateMemory(agent, memory, turnEvents) {
    if (!turnEvents || turnEvents.length === 0) {
      logger.info(`No turn events for ${agent.name}, skipping memory update.`);
      return memory; // No update needed
    }
    
    console.time(`[LLM] ${agent.name} memory update`);
    const loggingPayload = {
      type: 'memory_update',
      agent: agent.name,
      model: agent.model,
      input: {},
      output: null,
      error: null
    };

    try {
      // Check if memory has formatMemoryForLLM method
      if (typeof memory.formatMemoryForLLM !== 'function') {
        // Simple fallback if memory object doesn't have expected methods
        const memoryState = {
          knownInformation: memory.knownCards || {},
          currentDeductions: memory.currentMemory || "No current deductions available.",
          turnHistory: [] // Empty turn history
        };
        
        const turnEventsString = Array.isArray(turnEvents) ? turnEvents.join('\\n') : String(turnEvents);

        const userMessage = `You are ${agent.name}, playing Cluedo. Update your deductions based on the latest turn events. 
Your current knowledge:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
Current Memory: ${JSON.stringify(memoryState, null, 2)}

Events from the last turn:
${turnEventsString}

Analyze these events and update your deductions. What new facts are confirmed? What possibilities are eliminated? What seems more or less likely? Focus only on NEW insights derived *directly* from the turn events in the context of your existing knowledge. Be concise.

Respond with a short text summary of your new deductions or updated beliefs.`;

        loggingPayload.input = {
            prompt: userMessage,
            turnEvents: turnEventsString,
            initialMemoryState: memoryState
        };

        const completion = await cohere.chat({
          model: agent.model,
          message: userMessage,
          temperature: 0.2,
        });

        const llmDeductions = completion.text;
        loggingPayload.output = llmDeductions;

        // Since we don't have a memory.update method, just log the deductions
        logger.info(`Agent ${agent.name} memory update deductions: ${llmDeductions}`);
        
        await LoggingService.logLLMInteraction(loggingPayload);
        console.timeEnd(`[LLM] ${agent.name} memory update`);
        
        // Since we can't update memory properly, return the original
        return memory;
      }
      
      // Original implementation - used when memory has proper methods
      const memoryState = await memory.formatMemoryForLLM();
      const turnEventsString = Array.isArray(turnEvents) ? turnEvents.join('\\n') : String(turnEvents);

      const userMessage = `You are ${agent.name}, playing Cluedo. Update your deductions based on the latest turn events. 
Your current knowledge:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
Known Information: ${JSON.stringify(memoryState.knownInformation, null, 2)}
Current Deductions: ${memoryState.currentDeductions}

Events from the last turn:
${turnEventsString}

Analyze these events and update your deductions. What new facts are confirmed? What possibilities are eliminated? What seems more or less likely? Focus only on NEW insights derived *directly* from the turn events in the context of your existing knowledge. Be concise.

Respond with a short text summary of your new deductions or updated beliefs.`;

      loggingPayload.input = {
          prompt: userMessage,
          turnEvents: turnEventsString,
          initialMemoryState: memoryState
      };

      const completion = await cohere.chat({
        model: agent.model,
        message: userMessage,
        temperature: 0.2,
      });

      const llmDeductions = completion.text;
      loggingPayload.output = llmDeductions;

      if (!llmDeductions || typeof llmDeductions !== 'string') {
        logger.warn(`Invalid or empty response from LLM for memory update for ${agent.name}. Skipping update.`);
        loggingPayload.error = 'Invalid or empty response received';
        // Don't throw, just skip the update
      } else {
        // Pass the raw LLM deductions text to the memory update function
        if (typeof memory.update === 'function') {
          await memory.update(turnEventsString, llmDeductions);
        } else {
          // If memory.update doesn't exist, at least log the deductions
          logger.info(`Agent ${agent.name} memory update deductions: ${llmDeductions}`);
        }
      }
      
      await LoggingService.logLLMInteraction(loggingPayload);

      console.timeEnd(`[LLM] ${agent.name} memory update`);
      return memory; // Return the potentially updated memory object

    } catch (error) {
      logger.error(`Error in updateMemory for ${agent.name}: ${error.message}`, { stack: error.stack });
      console.timeEnd(`[LLM] ${agent.name} memory update`);

      loggingPayload.error = error.message;
      await LoggingService.logLLMInteraction(loggingPayload); // Log failure
      
      // Return original memory on error
      return memory; 
    }
  }

  /**
   * Evaluates which card an agent should show to challenge a suggestion, if any.
   *
   * @param {Agent} agent - The agent being challenged
   * @param {Object} suggestion - The suggestion being challenged {suspect, weapon, room}
   * @param {Array<string>} cards - Agent's cards that match the suggestion elements
   * @returns {Promise<{cardToShow: string|null, reasoning: string}>} The card to show (or null) and reasoning.
   */
  static async evaluateChallenge(agent, suggestion, cards) {
    // Ensure cards is always an array
    const cardArray = Array.isArray(cards) ? cards : (cards ? [cards] : []);
    
    if (!cardArray || cardArray.length === 0) {
        logger.info(`Agent ${agent.name} has no cards matching the suggestion.`);
        return { cardToShow: null, reasoning: "No matching cards held." };
    }

    console.time(`[LLM] ${agent.name} evaluate challenge`);
    const loggingPayload = {
        type: 'evaluate_challenge',
        agent: agent.name,
        model: agent.model,
        input: {},
        output: null,
        error: null,
        parsedOutput: null,
        validationStatus: 'pending'
    };

    try {
        // Replace the userMessage definition to ensure correct interpolation
        const userMessage = `You are ${agent.name}, playing Cluedo. You have been challenged based on the suggestion: ${suggestion.suspect} in the ${suggestion.room} with the ${suggestion.weapon}.

You hold the following card(s) that match the suggestion: ${cardArray.join(', ')}. 

Which card should you show? Consider:
1. Show only ONE card.
2. Prioritize showing a card that reveals the least information about the remaining solution (e.g., if you have multiple matching cards, maybe show one that others might already suspect you have, or one related to a less critical category based on your deductions).
3. If you only have one matching card, you must show it.

Respond ONLY with a JSON object in the following format:
{
  "cardToShow": "string (the name of the card to show, must be one of: ${cardArray.join(', ')})",
  "reasoning": "string (briefly explain your choice)"
}`; // Ensure both interpolations use .join(', ')

        loggingPayload.input = {
            prompt: userMessage,
            agentName: agent.name,
            suggestion,
            matchingCards: cardArray
        };

        // Check if the model supports response_format (Command R and newer)
        const supportsJsonResponseFormat = agent.model.startsWith('command-r');

        const apiParams = {
            model: agent.model,
            message: userMessage,
            temperature: 0.2,
        };

        if (supportsJsonResponseFormat) {
            apiParams.response_format = { type: "json_object" };
            logger.info(`[LLM Debug] Using response_format: json_object for model ${agent.model} in evaluateChallenge`);
        } else {
            logger.info(`[LLM Debug] Model ${agent.model} does not support response_format: json_object in evaluateChallenge. Relying on prompt.`);
        }

        let completion;
        try {
            console.log(`[LLMService DEBUG] Calling cohere.chat for evaluateChallenge with params:`, JSON.stringify(apiParams, null, 2)); // Log params
            completion = await cohere.chat(apiParams);
            console.log(`[LLMService DEBUG] cohere.chat call for evaluateChallenge completed successfully.`);
        } catch (apiError) {
            logger.error(`API call failed specifically within evaluateChallenge for ${agent.name} (Model: ${agent.model}):`, { 
                message: apiError.message, 
                statusCode: apiError.statusCode, // Log status code if available
                body: apiError.body, // Log body if available
                stack: apiError.stack 
            });
            loggingPayload.error = `API Call Error: ${apiError.message}`;
            loggingPayload.validationStatus = 'failed_api_call';
            // Ensure the outer try...catch knows an error occurred
            throw apiError; // Re-throw the error to be caught by the outer handler which returns null card
        }

        const responseText = completion.text;
        loggingPayload.output = responseText;
        console.log(`[LLMService DEBUG] Raw response from ${agent.name} (${agent.model}) evaluateChallenge:`, responseText);

        let parsedResult;
        try {
            // If response_format wasn't used, the text might contain markdown or just plain JSON
            // Use extractJSON which handles both cases
            if (!supportsJsonResponseFormat) {
                parsedResult = extractJSON(responseText);
            } else {
                // If response_format was used, the API should return clean JSON text
                parsedResult = JSON.parse(responseText);
            }

            if (!parsedResult) { // Check if parsing/extraction failed
                throw new Error('Failed to extract or parse JSON from response.');
            }
            loggingPayload.parsedOutput = parsedResult;
        } catch (parseError) {
            logger.error('Failed to parse challenge JSON response from Cohere:', { error: parseError.message, responseText });
            loggingPayload.error = `JSON Parsing Error: ${parseError.message}`;
            loggingPayload.validationStatus = 'failed_parsing';
            // Fallback: Return null if parsing fails
            logger.error('Fallback in evaluateChallenge due to parsing error. Returning null card.');
            return { cardToShow: null, reasoning: "LLM response parsing failed." };
        }

        // Basic validation
        if (!parsedResult || typeof parsedResult !== 'object' || typeof parsedResult.cardToShow !== 'string' || typeof parsedResult.reasoning !== 'string') {
            logger.error('Invalid structure in challenge JSON response:', { parsedResult });
            loggingPayload.error = 'Invalid JSON structure received';
            loggingPayload.validationStatus = 'failed_validation';
            // Fallback: Show the first card - Revisit: Should this also return null?
            logger.error('Fallback in evaluateChallenge due to invalid structure. Returning first card.'); 
            return { cardToShow: cardArray[0], reasoning: "Invalid JSON structure received, showing first available card." };
        }
        
        // Validate the chosen card is one the agent actually has and matches
        if (!cardArray.includes(parsedResult.cardToShow)) {
            logger.info(`LLM chose invalid card '${parsedResult.cardToShow}' for challenge. Agent holds: ${cardArray.join(', ')}. Forcing first valid card.`);
            loggingPayload.error = `Invalid card choice: ${parsedResult.cardToShow}`;
            loggingPayload.validationStatus = 'corrected_card';
            parsedResult.cardToShow = cardArray[0]; // Correct to the first valid card
            parsedResult.reasoning += " (Corrected: Invalid card chosen by LLM)";
        } else {
            loggingPayload.validationStatus = 'passed';
        }

        await LoggingService.logLLMInteraction(loggingPayload);

        console.timeEnd(`[LLM] ${agent.name} evaluate challenge`);
        return parsedResult;

    } catch (error) {
        logger.error(`Error in evaluateChallenge for ${agent.name}: ${error.message}`, { stack: error.stack });
        console.timeEnd(`[LLM] ${agent.name} evaluate challenge`);
        
        loggingPayload.error = error.message;
        loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;
        await LoggingService.logLLMInteraction(loggingPayload);

        throw new Error(`LLM challenge evaluation failed for ${agent.name}: ${error.message}`);
    }
  }

  /**
   * Determines if the agent should make an accusation based on its confidence.
   *
   * @param {Agent} agent - The agent considering the accusation
   * @param {Object} gameState - Current game state information
   * @returns {Promise<Object>} Accusation decision object including boolean `shouldAccuse`, 
   *                            the `accusation` {suspect, weapon, room}, 
   *                            `confidence` {suspect, weapon, room}, and `reasoning`.
   */
  static async considerAccusation(agent, gameState) {
    console.time(`[LLM] ${agent.name} consider accusation`);
    const loggingPayload = {
        type: 'consider_accusation',
        agent: agent.name,
        model: agent.model,
        input: {},
        output: null,
        error: null,
        parsedOutput: null,
        validationStatus: 'pending'
    };

    try {
      const memoryState = await agent.memory.formatMemoryForLLM();
      
      const userMessage = `You are ${agent.name}, playing Cluedo. Analyze your memory and decide if you should make a final accusation.

Your current knowledge:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
Known Information: ${JSON.stringify(memoryState.knownInformation, null, 2)}
Current Deductions: ${memoryState.currentDeductions}
Recent Turn History:
${memoryState.turnHistory.join('\\n')}

Consider the following:
1. How certain are you about the suspect, weapon, AND room?
2. Accusing incorrectly means you lose the game immediately.
3. Only accuse if you are highly confident (e.g., >90-95% certainty) in all three elements.

Respond ONLY with a JSON object in the following format. 
- If you decide NOT to accuse, set \"shouldAccuse\" to false and provide nulls for the accusation details, but still estimate your confidence levels.
- If you decide TO accuse, set \"shouldAccuse\" to true and fill in the suspect, weapon, and room you are accusing.

{
  "shouldAccuse": boolean,
  "accusation": {
    "suspect": string | null, // (Provide value if shouldAccuse is true, otherwise null)
    "weapon": string | null,  // (Provide value if shouldAccuse is true, otherwise null)
    "room": string | null     // (Provide value if shouldAccuse is true, otherwise null)
  },
  "confidence": { // Estimate your confidence (0.0 to 1.0) for each element REGARDLESS of accusation decision
    "suspect": number (0.0-1.0),
    "weapon": number (0.0-1.0),
    "room": number (0.0-1.0)
  },
  "reasoning": string (Explain your decision and confidence assessment briefly)
}`;

      loggingPayload.input = {
          prompt: userMessage,
          agentName: agent.name,
          memoryState // Log memory state used for decision
      };

      console.log(`[LLMService DEBUG] Attempting cohere.chat for ${agent.name} (Model: ${agent.model}) accusation...`);

      // Check if the model supports response_format (Command R and newer)
      const supportsJsonResponseFormat = agent.model.startsWith('command-r');

      const apiParams = {
        model: agent.model,
        message: userMessage,
        temperature: 0.1,
      };

      if (supportsJsonResponseFormat) {
        apiParams.response_format = { type: "json_object" };
        logger.info(`[LLM Debug] Using response_format: json_object for model ${agent.model} in considerAccusation`);
      } else {
        logger.info(`[LLM Debug] Model ${agent.model} does not support response_format: json_object in considerAccusation. Relying on prompt.`);
      }

      const completion = await cohere.chat(apiParams);
      
      console.log(`[LLMService DEBUG] cohere.chat completed for ${agent.name} (Model: ${agent.model}) accusation.`);

      const responseText = completion.text;
      loggingPayload.output = responseText;

      // Use safeParseJSON which includes schema validation and confidence normalization
      const { valid, data: parsedResult, error: validationError } = safeParseJSON(responseText, accusationSchema);

      if (!valid) {
        logger.error('Accusation response failed validation:', { validationError, responseText });
        loggingPayload.error = `Validation Failed: ${JSON.stringify(validationError)}`;
        loggingPayload.validationStatus = 'failed_validation';
        // Fallback: Do not accuse if validation fails
        return {
          shouldAccuse: false,
          accusation: { suspect: null, weapon: null, room: null },
          confidence: { suspect: 0, weapon: 0, room: 0 },
          reasoning: "LLM response failed validation, choosing not to accuse."
        };
      }

      loggingPayload.parsedOutput = parsedResult; // Log the validated and normalized data
      loggingPayload.validationStatus = 'passed';
      await LoggingService.logLLMInteraction(loggingPayload);

      console.timeEnd(`[LLM] ${agent.name} consider accusation`);
      return parsedResult;

    } catch (error) {
      logger.error(`Error in considerAccusation for ${agent.name}: ${error.message}`, { stack: error.stack });
      console.timeEnd(`[LLM] ${agent.name} consider accusation`);

      loggingPayload.error = error.message;
      loggingPayload.validationStatus = loggingPayload.validationStatus === 'pending' ? 'failed_api_call' : loggingPayload.validationStatus;
      await LoggingService.logLLMInteraction(loggingPayload);
      
      // Fallback strategy on API error: Do not accuse
      logger.info(`LLM considerAccusation failed for ${agent.name}. Defaulting to not accusing.`);
      return {
        shouldAccuse: false,
        accusation: { suspect: null, weapon: null, room: null },
        confidence: { suspect: 0, weapon: 0, room: 0 },
        reasoning: "LLM API error, choosing not to accuse."
      };
    }
  }
}