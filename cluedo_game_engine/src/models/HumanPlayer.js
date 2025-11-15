import { Memory } from './Memory.js';
import { logger } from '../utils/logger.js';

/**
 * Represents a human player in the Cluedo game.
 * Unlike AI agents, this class waits for user input via Socket.IO events
 * instead of calling LLM services.
 */
export class HumanPlayer {
  /**
   * Creates a new HumanPlayer instance.
   *
   * @param {string} name - The player's name (e.g., "You", "Human Player")
   * @param {Array<string>} cards - Cards in the player's hand
   * @param {Object} gameRef - Reference to the game instance
   * @param {Object} io - Socket.IO instance for communication
   * @param {string} socketId - Socket ID of the connected player
   */
  constructor(name, cards, gameRef, io, socketId) {
    this.name = name;
    this.cards = new Set(cards);
    this.model = 'human'; // Special identifier for human players
    this.hasLost = false;
    this.game = gameRef;
    this.io = io;
    this.socketId = socketId;

    // Initialize memory system (same as AI agents)
    this.memory = new Memory(name);
    this.memory.game = gameRef;

    // Initialize with known cards
    cards.forEach(card => this.memory.addKnownCard(card));

    // Location tracking
    this.location = null;

    logger.info(`Human player ${name} created with ${cards.length} cards`);
  }

  /**
   * Moves the human player to a room.
   * For simplicity, randomly selects a room (could be enhanced to let user choose).
   *
   * @param {Array<string>} availableRooms - List of available rooms
   */
  move(availableRooms) {
    if (!this.location || Math.random() > 0.5) {
      this.location = availableRooms[Math.floor(Math.random() * availableRooms.length)];
    }
    logger.info(`${this.name} moved to ${this.location}`);
  }

  /**
   * Waits for the human player to make a suggestion via the UI.
   * Emits a request to the client and returns a Promise that resolves
   * when the player submits their suggestion.
   *
   * @param {Object} gameState - Current game state
   * @returns {Promise<Object>} Suggestion with suspect, weapon, room, and reasoning
   */
  async makeSuggestion(gameState) {
    logger.info(`Waiting for ${this.name} to make a suggestion...`);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Human player suggestion timed out'));
      }, 300000); // 5 minute timeout

      // Send request to client
      this.io.emit('request-suggestion', {
        playerName: this.name,
        location: this.location,
        cards: Array.from(this.cards),
        gameState: {
          currentTurn: gameState.currentTurn,
          availableSuspects: gameState.availableSuspects,
          availableWeapons: gameState.availableWeapons,
          availableRooms: gameState.availableRooms,
        },
        memory: {
          knownCards: Array.from(this.memory.knownCards),
          eliminatedCards: Array.from(this.memory.eliminatedCards),
        },
      });

      // Set up one-time listener for response
      const responseHandler = suggestion => {
        clearTimeout(timeout);
        this.io.off('human-suggestion', responseHandler);

        // Validate suggestion
        if (!suggestion || !suggestion.suspect || !suggestion.weapon || !suggestion.room) {
          reject(new Error('Invalid suggestion from human player'));
          return;
        }

        // Ensure room matches location
        if (suggestion.room !== this.location) {
          logger.warn(
            `Human player suggested room (${suggestion.room}) doesn't match location (${this.location}). Correcting.`
          );
          suggestion.room = this.location;
        }

        logger.info(`${this.name} suggested: ${JSON.stringify(suggestion)}`);
        resolve(suggestion);
      };

      this.io.once('human-suggestion', responseHandler);
    });
  }

  /**
   * Waits for the human player to decide whether to make an accusation.
   *
   * @param {Object} suggestion - The suggestion made this turn
   * @param {Object} challengeResult - Result of challenges
   * @returns {Promise<Object>} Accusation decision
   */
  async considerAccusation(suggestion, challengeResult) {
    logger.info(`Waiting for ${this.name} to consider accusation...`);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        // Auto-resolve to not accuse if timeout
        resolve({
          shouldAccuse: false,
          accusation: { suspect: null, weapon: null, room: null },
          reasoning: 'Timed out, defaulting to no accusation',
        });
      }, 60000); // 1 minute timeout for accusation decision

      // Send request to client
      this.io.emit('request-accusation', {
        playerName: this.name,
        suggestion,
        challengeResult,
        memory: {
          knownCards: Array.from(this.memory.knownCards),
          eliminatedCards: Array.from(this.memory.eliminatedCards),
        },
      });

      // Set up one-time listener for response
      const responseHandler = decision => {
        clearTimeout(timeout);
        this.io.off('human-accusation', responseHandler);

        logger.info(`${this.name} accusation decision: ${JSON.stringify(decision)}`);
        resolve(decision);
      };

      this.io.once('human-accusation', responseHandler);
    });
  }

  /**
   * Waits for the human player to choose which card to show in response to a challenge.
   *
   * @param {Object} suggestion - The suggestion being challenged
   * @returns {Promise<Object|null>} Card to show or null if no cards match
   */
  async respondToChallenge(suggestion) {
    const matchingCards = Array.from(this.cards).filter(
      card =>
        card === suggestion.suspect || card === suggestion.weapon || card === suggestion.room
    );

    if (matchingCards.length === 0) {
      return null; // Cannot challenge
    }

    logger.info(`Waiting for ${this.name} to choose card to show...`);

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        // Auto-select first matching card if timeout
        resolve({
          cardToShow: matchingCards[0],
          reasoning: 'Timed out, auto-selected first matching card',
        });
      }, 60000); // 1 minute timeout

      // Send request to client
      this.io.emit('request-challenge-response', {
        playerName: this.name,
        suggestion,
        matchingCards,
      });

      // Set up one-time listener for response
      const responseHandler = response => {
        clearTimeout(timeout);
        this.io.off('human-challenge-response', responseHandler);

        // Validate response
        if (!matchingCards.includes(response.cardToShow)) {
          logger.warn(
            `Human player chose invalid card ${response.cardToShow}. Using first matching card.`
          );
          response.cardToShow = matchingCards[0];
        }

        logger.info(`${this.name} chose to show: ${response.cardToShow}`);
        resolve(response);
      };

      this.io.once('human-challenge-response', responseHandler);
    });
  }

  /**
   * Marks the human player as eliminated.
   */
  setLost() {
    this.hasLost = true;
    this.memory.currentMemory += '\n\n[ELIMINATED] Made an incorrect accusation and was eliminated from the game.';
    logger.info(`${this.name} has been eliminated`);

    // Notify client
    if (this.io && this.socketId) {
      this.io.emit('player-eliminated', {
        message: 'You made an incorrect accusation and have been eliminated from the game.',
      });
    }
  }

  /**
   * Returns whether the player is human.
   * @returns {boolean}
   */
  isHuman() {
    return true;
  }
}
