import { Memory } from './Memory.js';
import { LLMService } from '../services/llm.js';

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

      return await LLMService.makeSuggestion(this, validGameState);
    } catch (error) {
      console.error(`${this.name} failed to make suggestion:`, error);
      // Return fallback suggestion
      return {
        suspect: validGameState.availableSuspects[0],
        weapon: validGameState.availableWeapons[0],
        room: validGameState.availableRooms[0],
        reasoning: 'Error occurred, using fallback suggestion'
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
      const result = await LLMService.evaluateChallenge(this, suggestion, cardsArray);
      
      // Update memory with challenge result
      if (result.cardToShow) {
        this.memory.addKnownCard(result.cardToShow);
      }
      
      return result;
    } catch (error) {
      console.error(`Challenge evaluation failed for ${this.name}:`, error);
      // If evaluation fails, don't show any card
      return { cardToShow: null, reasoning: "Error in challenge evaluation" };
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
        console.error(`Game reference missing for ${this.name}`);
        throw new Error('Game reference not set');
      }
      
      console.log(`[DEBUG] Formatting memory for ${this.name} accusation consideration...`);
      const memoryState = await this.memory.formatMemoryForLLM();
      console.log(`[DEBUG] Memory formatted for ${this.name}.`);

      const gameState = {
        currentTurn: this.game.currentTurn,
        knownCards: this.cards,
        memory: memoryState
      };
      
      console.log(`[DEBUG] Calling LLMService.considerAccusation for ${this.name} (Model: ${this.model})...`);
      const result = await LLMService.considerAccusation(this, gameState);
      console.log(`[DEBUG] LLMService.considerAccusation completed for ${this.name}.`);
      return result;

    } catch (error) {
      console.error(`Accusation consideration failed for ${this.name}:`, error);
      return {
        shouldAccuse: false,
        accusation: { suspect: null, weapon: null, room: null },
        confidence: { suspect: 0, weapon: 0, room: 0 },
        reasoning: 'Error in accusation consideration'
      };
    }
  }
  

  /**
   * Updates the agent's memory with new information from a turn.
   * 
   * This method:
   * 1. Gets the LLM's interpretation of the turn events
   * 2. Updates the agent's memory state with new information
   * 3. Records the turn in memory history
   * 
   * @param {Object} turnEvents - Events from the current turn
   * @returns {Promise<void>}
   */
  async updateMemory(turnEvents) {
    try {
      // Get LLM's interpretation of the turn
      const updatedMemory = await LLMService.updateMemory(
        this,
        this.memory.currentMemory,
        turnEvents
      );

      // Update memory state
      this.memory.currentMemory = updatedMemory;
      
      // Update memory with turn events (this now handles history management internally)
      await this.memory.updateMemory(turnEvents);

    } catch (error) {
      console.error(`Failed to update ${this.name}'s memory:`, error);
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
      console.log(`${this.name} assigned starting location: ${this.location}`); // Log initial assignment
    } else {
      // In a full implementation, add logic for moving to adjacent rooms.
      // For now, the agent stays put after the initial assignment.
      console.log(`${this.name} stays in location: ${this.location}`);
    }
  }
} 