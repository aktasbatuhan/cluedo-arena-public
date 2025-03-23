import OpenAI from "openai";
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
 * All models are accessed through OpenRouter API.
 * 
 * @type {Array<string>}
 */
export const MODEL_LIST = [
  'mistralai/mistral-small-3.1-24b-instruct',                             
  'anthropic/claude-3.5-sonnet',           
  'google/gemini-2.0-flash-001',          
  'cohere/command-a',                         
  'openai/gpt-4o',   
  'anthropic/claude-3.5-sonnet'          
];

// Initialize OpenAI client with OpenRouter configuration
const openai = new OpenAI({
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    "HTTP-Referer": process.env.SITE_URL,
    "X-Title": process.env.SITE_NAME,
  },
  timeout: 30000
});

// Initialize JSON schema validator
const ajv = new Ajv();

// Define JSON schemas
const suggestionSchema = {
  type: "object",
  properties: {
    suspect: { type: "string" },
    weapon: { type: "string" },
    room: { type: "string" },
    reasoning: { type: "string" }
  },
  required: ["suspect", "weapon", "room"],
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
    console.error('JSON extraction failed:', error.message);
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
    try {
      const memoryState = await agent.memory.formatMemoryForLLM();
      
      const prompt = `As ${agent.name}, analyze the game state:
Known cards: ${Array.from(agent.cards).join(', ')}
Current turn: ${gameState.currentTurn}
Available suspects: ${gameState.availableSuspects.join(', ')}
Available weapons: ${gameState.availableWeapons.join(', ')}
Available rooms: ${gameState.availableRooms.join(', ')}

Your memory and deductions:
${JSON.stringify(memoryState.knownInformation, null, 2)}

Current deductions:
${memoryState.currentDeductions}

Recent turn history:
${memoryState.turnHistory.join('\n')}

Make a strategic suggestion considering:
1. Your known cards and deductions
2. Previous suggestions and their outcomes
3. Your current memory state
4. Strategic room positioning

Format response as JSON:
{
  "suspect": "string (must be from available suspects)",
  "weapon": "string (must be from available weapons)",
  "room": "string (must be from available rooms)",
  "reasoning": "string explaining your strategy"
}
Only return the JSON following the format above, nothing else.`;

      const completion = await openai.chat.completions.create({
        model: agent.model,
        messages: [
          {
            role: "system",
            content: "You are a strategic Cluedo/Clue player. Make logical deductions and strategic suggestions."
          },
          { role: "user", content: prompt }
        ]
      });

      const response = completion.choices[0].message.content;
      
      // Log the interaction
      await LoggingService.logLLMInteraction({
        type: 'suggestion',
        agent: agent.name,
        model: agent.model,
        input: {
          prompt,
          gameState: {
            knownCards: Array.from(agent.cards),
            currentTurn: gameState.currentTurn,
            recentHistory: gameState.recentHistory
          }
        },
        output: response
      });

      // Parse and validate with schema
      const parsedResult = extractJSON(response);
      if (!parsedResult) {
        throw new Error('Failed to parse suggestion response');
      }
      
      // Validate against schema
      const validate = ajv.compile(suggestionSchema);
      if (!validate(parsedResult)) {
        console.error('Suggestion validation errors:', validate.errors);
        throw new Error('Invalid suggestion format: ' + JSON.stringify(validate.errors));
      }
      
      // Validate that values are from available options
      if (!gameState.availableSuspects.includes(parsedResult.suspect)) {
        throw new Error(`Invalid suspect: ${parsedResult.suspect} is not in available suspects`);
      }
      if (!gameState.availableWeapons.includes(parsedResult.weapon)) {
        throw new Error(`Invalid weapon: ${parsedResult.weapon} is not in available weapons`);
      }
      if (!gameState.availableRooms.includes(parsedResult.room)) {
        throw new Error(`Invalid room: ${parsedResult.room} is not in available rooms`);
      }

      console.timeEnd(`[LLM] ${agent.name} suggestion`);
      return {
        suspect: parsedResult.suspect,
        weapon: parsedResult.weapon,
        room: parsedResult.room,
        reasoning: parsedResult.reasoning || 'No reasoning provided'
      };

    } catch (error) {
      console.timeEnd(`[LLM] ${agent.name} suggestion`);
      console.error('Suggestion generation failed:', error);
      // Return a fallback suggestion
      return {
        suspect: gameState.availableSuspects[0],
        weapon: gameState.availableWeapons[0],
        room: gameState.availableRooms[0],
        reasoning: 'Error occurred, using fallback suggestion'
      };
    }
  }
  

  static async updateMemory(agent, memory, turnEvents) {
    console.time(`[LLM] ${agent.name} memory update`);
    try {
      const prompt = `As ${agent.name}, analyze this complete turn:

Turn Number: ${turnEvents.turnNumber}
Active Agent: ${turnEvents.activeAgent}

Suggestion Made:
${turnEvents.suggestion ? 
  `${turnEvents.activeAgent} suggested ${turnEvents.suggestion.suspect} in the ${turnEvents.suggestion.room} with the ${turnEvents.suggestion.weapon}` 
  : 'No suggestion made'}

Challenge Result:
${turnEvents.challengeResult?.canChallenge ? 
  `${turnEvents.challengeResult.challengingAgent} showed a card to disprove the suggestion` 
  : 'No successful challenge'}

Current Memory State: ${memory}

Update your memory with all this turn's information...`;

      // Get LLM's interpretation and deductions
      const completion = await openai.chat.completions.create({
        model: agent.model,
        messages: [
          { role: "system", content: "You are a Cluedo player updating your memory." },
          { role: "user", content: prompt }
        ]
      });

      // Add null check for completion data
      if (!completion?.choices?.[0]?.message?.content) {
        console.error('Invalid response format from API');
        return memory; // Return original memory if update fails
      }

      const response = completion.choices[0].message.content;

      // Log the interaction
      await LoggingService.logLLMInteraction({
        type: 'memory_update',
        agent: agent.name,
        model: agent.model,
        turn: turnEvents.turnNumber,
        input: {
          prompt,
          currentMemory: memory,
          turnEvents
        },
        output: response
      });

      console.timeEnd(`[LLM] ${agent.name} memory update`);
      return response;
    } catch (error) {
      console.timeEnd(`[LLM] ${agent.name} memory update`);
      console.error(`LLM memory update error for ${agent.name}:`, error);
      throw error;
    }
  }

  static async evaluateChallenge(agent, suggestion, cards) {
    const timeLabel = `[LLM] ${agent.name} challenge evaluation`;
    console.time(timeLabel);
    try {
      const matchingCards = Array.from(cards).filter(card => 
        card === suggestion.suspect ||
        card === suggestion.weapon ||
        card === suggestion.room
      );

      if (matchingCards.length === 0) {
        console.timeEnd(timeLabel);
        return {
          canChallenge: false,
          cardToShow: null
        };
      }

      // If only one matching card, show that
      if (matchingCards.length === 1) {
        console.timeEnd(timeLabel);
        return {
          canChallenge: true,
          cardToShow: matchingCards[0]
        };
      }

      // If multiple matching cards, let LLM choose strategically
      const prompt = `As ${agent.name}, you need to choose which card to show to disprove a suggestion.
You have these matching cards: ${matchingCards.join(', ')}
The suggestion was: ${suggestion.suspect} in the ${suggestion.room} with the ${suggestion.weapon}

Choose one card to show based on strategic value.
Format response as JSON:
{
  "cardToShow": "string (must be one of your matching cards)"
}

Only return the JSON following the format above, nothing else.`;

      const completion = await openai.chat.completions.create({
        model: agent.model,
        messages: [
          { 
            role: "system", 
            content: "You are a strategic Cluedo/Clue player choosing which card to show to disprove a suggestion."
          },
          { role: "user", content: prompt }
        ]
      });

      const response = completion.choices[0].message.content;
      const result = extractJSON(response);

      if (!result || !matchingCards.includes(result.cardToShow)) {
        console.timeEnd(timeLabel);
        return {
          canChallenge: true,
          cardToShow: matchingCards[0]  // Fallback to first matching card if LLM response is invalid
        };
      }

      console.timeEnd(timeLabel);
      return {
        canChallenge: true,
        cardToShow: result.cardToShow
      };

    } catch (error) {
      console.timeEnd(timeLabel);
      console.error('Challenge evaluation failed:', error);
      // Fallback to simple challenge if error occurs
      const matchingCard = Array.from(cards).find(card => 
        card === suggestion.suspect ||
        card === suggestion.weapon ||
        card === suggestion.room
      );
      
      return {
        canChallenge: !!matchingCard,
        cardToShow: matchingCard || null
      };
    }
  }

  static async considerAccusation(agent, gameState) {
    console.time(`[LLM] ${agent.name} accusation consideration`);
    try {
      const memoryState = await agent.memory.formatMemoryForLLM();
      
      const prompt = `As ${agent.name}, analyze the game state:
Known cards: ${Array.from(agent.cards).join(', ')}
Memory state: ${JSON.stringify(memoryState)}

Based on your suggestion this turn and the challenge results:
1. What cards were shown/not shown?
2. What can you deduce about card locations?
3. How confident are you about the solution?

Should you make an accusation? Consider:
1. Confidence in each element (suspect, weapon, room)
2. Risk of being eliminated if wrong
3. Information gained from your suggestion and the challenge

Format response as JSON:
{
  "shouldAccuse": boolean,
  "accusation": {
    "suspect": "string",
    "weapon": "string",
    "room": "string"
  },
  "confidence": {
    "suspect": number (0-1),
    "weapon": number (0-1),
    "room": number (0-1)
  },
  "reasoning": "string"
}

Only return the JSON following the format above, nothing else.`;

      const completion = await openai.chat.completions.create({
        model: agent.model,
        messages: [
          {
            role: "system",
            content: "You are a strategic Cluedo/Clue player deciding whether to make an accusation."
          },
          { role: "user", content: prompt }
        ]
      });

      const response = completion.choices[0].message.content;
      const result = extractJSON(response);

      if (!result) {
        throw new Error('Failed to parse accusation response');
      }

      console.timeEnd(`[LLM] ${agent.name} accusation consideration`);
      return result;

    } catch (error) {
      console.timeEnd(`[LLM] ${agent.name} accusation consideration`);
      console.error('Accusation consideration failed:', error);
      return {
        shouldAccuse: false,
        accusation: { suspect: null, weapon: null, room: null },
        confidence: { suspect: 0, weapon: 0, room: 0 },
        reasoning: 'Error in accusation consideration'
      };
    }
  }
} 