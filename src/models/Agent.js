import { Memory } from './Memory.js';
import { LLMService } from '../services/llm.js';
import { logger } from '../utils/logger.js';

/**
 * Represents an AI agent in the Cluedo game.
 * 
 * Each Agent:
 * - Holds its own cards
 * - Maintains a memory of game events
 * - Interacts with an LLM to make decisions
 * - Can make suggestions, respond to challenges, and consider accusations
 */
export class Agent {
  /**
   * Creates a new Agent instance.
   * 
   * @param {string} name - The agent's name/identifier
   * @param {Array<string>} cards - Cards in the agent's hand
   * @param {string} model - The LLM model to use for this agent
   * @param {string|Object} gameId - Game identifier or reference
   */
  constructor(name, cards, model, gameId) {
    this.name = name;
    this.cards = new Set(cards); // Ensure cards is a Set
    this.model = model;
    this.hasLost = false;
    this.game = null; // Will be set by the Game class
    
    // Initialize memory system
    this.memory = new Memory(gameId);
    
    // Initialize with known cards
    cards.forEach(card => this.memory.addKnownCard(card));
    
    // Temporary storage for request IDs from LLMService calls
    this.lastSuggestionRequestId = null;
    this.lastMemoryUpdateRequestId = null;
    this.lastChallengeRequestId = null;
    this.lastAccusationRequestId = null;

    // Validate memory system integrity
    if (typeof this.memory.formatMemoryForLLM !== 'function') {
      throw new Error('Memory system corrupted - missing format method');
    }
  }

  /**
   * Makes a strategic suggestion during the agent's turn.
   * 
   * @param {Object} gameState - Current game state information
   * @returns {Promise<Object>} Suggestion with suspect, weapon, room and reasoning
   */
  async makeSuggestion(gameState) {
    try {
      // Ensure gameState has required properties
      const validGameState = {
        currentTurn: gameState.currentTurn || 0,
        availableSuspects: gameState.availableSuspects || [],
        availableWeapons: gameState.availableWeapons || [],
        availableRooms: gameState.availableRooms || [],
        recentHistory: gameState.recentHistory || [],
        activePlayers: gameState.activePlayers || 0
      };

      // Validate cards is a Set
      if (!(this.cards instanceof Set)) {
        this.cards = new Set(this.cards);
      }

      // Call LLMService, which now returns result + requestId
      const result = await LLMService.makeSuggestion(this, validGameState);
      
      // Store the request ID for this specific action (use nullish coalescing)
      this.lastSuggestionRequestId = result.requestId ?? null;
      
      // Return only the suggestion part (or error if present)
      if (result.error) {
          // If LLMService returned a fallback object with an error field
          return { 
              suspect: result.suspect, 
              weapon: result.weapon, 
              room: result.room, 
              reasoning: result.reasoning, 
              error: result.error 
          };
      } else {
          return { 
              suspect: result.suspect, 
              weapon: result.weapon, 
              room: result.room, 
              reasoning: result.reasoning 
          };
      }
      
    } catch (error) {
      logger.error(`${this.name} agent-level error during makeSuggestion: ${error.message}`, { error });
      this.lastSuggestionRequestId = null; // Clear ID on error
      // Return fallback suggestion
      return {
        suspect: gameState.availableSuspects ? gameState.availableSuspects[0] : 'Miss Scarlet',
        weapon: gameState.availableWeapons ? gameState.availableWeapons[0] : 'Candlestick',
        room: this.location || 'Lounge', // Use current location or default
        reasoning: `Agent error occurred, using fallback suggestion: ${error.message}`,
        error: error.message
      };
    }
  }

  /**
   * Evaluates a suggestion from another agent and determines which card to show.
   * 
   * When this agent has one or more cards mentioned in another agent's suggestion,
   * this method decides which card to strategically reveal to disprove the suggestion.
   * 
   * @param {Object} suggestion - The suggestion to evaluate (suspect, weapon, room)
   * @returns {Promise<Object>} Challenge result with canChallenge flag and cardToShow
   */
  async evaluateChallenge(suggestion, matchingCards) {
    // Ensure matchingCards is an array of strings, not a Set or single value
    const cardsArray = matchingCards ? 
      (Array.isArray(matchingCards) ? matchingCards : [matchingCards]) : 
      [];
    
    try {
      // Call LLMService, which returns result + requestId
      const result = await LLMService.evaluateChallenge(this, suggestion, cardsArray);
      
      // Store the request ID (use nullish coalescing)
      this.lastChallengeRequestId = result.requestId ?? null;
      
      // Update memory with challenge result if successful and card shown
      if (result.cardToShow && !result.error) {
        this.memory.addKnownCard(result.cardToShow);
      }
      
      // Return only the challenge result part (or error)
      return { cardToShow: result.cardToShow, reasoning: result.reasoning, error: result.error || null };

    } catch (error) {
      logger.error(`Agent-level challenge evaluation failed for ${this.name}: ${error.message}`, { error });
      this.lastChallengeRequestId = null;
      // If evaluation fails, don't show any card
      return { cardToShow: null, reasoning: `Agent error in challenge evaluation: ${error.message}`, error: error.message };
    }
  }

  /**
   * Determines whether to make an accusation and with what confidence.
   * 
   * Based on all information gathered through suggestions, challenges,
   * and memory, this method decides if the agent should make a final
   * accusation about the solution.
   * 
   * @returns {Promise<Object>} Decision object containing:
   *                           - shouldAccuse: Boolean
   *                           - accusation: {suspect, weapon, room}
   *                           - confidence: Confidence levels for each element
   *                           - reasoning: Explanation of decision
   */
  async considerAccusation() {
    try {
      // Ensure game reference exists
      if (!this.game) {
        logger.error(`Game reference missing for ${this.name}`);
        throw new Error('Game reference not set');
      }
      
      logger.debug(`Formatting memory for ${this.name} accusation consideration...`);
      const memoryState = await this.memory.formatMemoryForLLM();
      logger.debug(`Memory formatted for ${this.name}.`);

      const gameState = {
        currentTurn: this.game.currentTurn,
        knownCards: this.cards,
        memory: memoryState
      };
      
      logger.debug(`Calling LLMService.considerAccusation for ${this.name} (Model: ${this.model})...`);
      // Call LLMService, which returns result + requestId
      const result = await LLMService.considerAccusation(this, gameState);
      logger.debug(`LLMService.considerAccusation completed for ${this.name}.`);
      
      // Store request ID (use nullish coalescing)
      this.lastAccusationRequestId = result.requestId ?? null;
      
      // Return only the decision part (or error)
      return { 
          shouldAccuse: result.shouldAccuse, 
          accusation: result.accusation, 
          confidence: result.confidence, 
          reasoning: result.reasoning, 
          error: result.error || null 
      };

    } catch (error) {
      logger.error(`Agent-level accusation consideration failed for ${this.name}: ${error.message}`, { error });
      this.lastAccusationRequestId = null;
      return {
        shouldAccuse: false,
        accusation: { suspect: null, weapon: null, room: null },
        confidence: { suspect: 0, weapon: 0, room: 0 },
        reasoning: `Agent error in accusation consideration: ${error.message}`,
        error: error.message
      };
    }
  }
  

  /**
   * Updates the agent's memory with events from the most recent turn.
   * 
   * @param {Array} turnEvents - Array of events from the most recent turn
   * @returns {Promise<{memory: Memory, deducedCards: Array<string>}>} Updated memory and array of deduced cards
   */
  async updateMemory(turnEvents) {
    try {
      // Log the turnEvents received by the agent before calling the service
      logger.debugObj(`[Agent.js Debug - ${this.name}] Received turnEvents for LLMService:`, turnEvents);
      
      // Call LLMService, which returns result + requestId
      const updateResult = await LLMService.updateMemory(this, this.memory, turnEvents);
      
      // Store the request ID for this memory update (use nullish coalescing)
      this.lastMemoryUpdateRequestId = updateResult.requestId ?? null;
      
      // LLMService.updateMemory already updates the memory object internally if memory.update exists.
      // We just need to return the structured result (without request ID) for Game.js
      return { 
          deducedCards: updateResult.deducedCards, 
          summary: updateResult.summary, 
          error: updateResult.error || null // Pass along potential errors from LLMService
      };

    } catch (error) {
      logger.error(`Agent-level error during memory update for ${this.name}: ${error.message}`, { error });
      this.lastMemoryUpdateRequestId = null;
      // Return empty/error results on error
      return { 
          deducedCards: [], 
          summary: `(Agent error in memory update: ${error.message})`, 
          error: error.message 
      };
    }
  }

  /**
   * Marks this agent as having lost the game.
   * Updates memory to reflect that an incorrect accusation was made.
   */
  setLost() {
    this.hasLost = true;
    this.memory.currentMemory += "\nI made an incorrect accusation and lost the game.";
  }

  /**
   * Gets a formatted representation of this agent's memory state.
   * 
   * @returns {Object} The agent's current memory and model
   */
  getMemoryState() {
    return {
      ...this.memory.formatMemoryForLLM(),
      model: this.model
    };
  }

  /**
   * Processes this agent's turn.
   * Performs maintenance operations on memory.
   * 
   * @returns {Promise<void>}
   */
  async processTurn() {
    // Perform memory maintenance (cleanup, optimization)
    this.memory.maintain();
  }

  /**
   * Moves the agent to a new location (room).
   * For now, this is a placeholder. In a real game, this would involve
   * dice rolls, pathfinding, or player choice.
   * It ensures the agent has *some* location assigned.
   *
   * @param {Array<string>} availableRooms - List of all possible rooms.
   */
  move(availableRooms) {
    // Simple placeholder: Assign a random room if no location exists
    // Or potentially move to an adjacent room in a full implementation.
    if (!this.location) {
      this.location = availableRooms[Math.floor(Math.random() * availableRooms.length)];
      logger.debug(`${this.name} assigned starting location: ${this.location}`);
    } else {
      // In a full implementation, add logic for moving to adjacent rooms.
      // For now, the agent stays put after the initial assignment.
      logger.debug(`${this.name} stays in location: ${this.location}`);
    }
  }
} 