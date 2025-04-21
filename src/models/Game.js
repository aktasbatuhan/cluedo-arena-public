import { LLMService } from '../services/llm.js';
import { Agent } from './Agent.js';
import { GameResult } from './GameResult.js';
import { MODEL_LIST } from '../services/llm.js';
import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { LoggingService } from '../services/LoggingService.js';
import { logger } from '../utils/logger.js';
import axios from 'axios';

let ioInstance = null;

// --- Configuration ---
const ART_WRAPPER_URL = process.env.ART_WRAPPER_URL || 'http://localhost:5001'; // Default wrapper URL

/**
 * Represents a game of Cluedo/Clue with AI agents powered by LLMs.
 * Manages the game state, turn progression, and interactions between agents.
 * @extends EventEmitter
 */
export class Game extends EventEmitter {
  /**
   * Creates a new Cluedo game instance.
   * @param {string} mode - Game mode ('spectate' or 'play')
   * @param {object} io - Socket.io instance for real-time communication
   */
  constructor(mode = 'spectate', io = null) {
    super(); // Initialize EventEmitter
    this.logger = logger; // Initialize the logger instance property
    this.io = io;
    // Traditional Cluedo suspects (not agent names)
    this.SUSPECTS = [
      'Miss Scarlet',
      'Colonel Mustard',
      'Mrs. White',
      'Mr. Green',
      'Mrs. Peacock',
      'Professor Plum'
    ];

    // Distinct agent names (colors)
    this.AGENT_NAMES = [
      'Red Agent',
      'Blue Agent',
      'Green Agent',
      'Yellow Agent',
      'Purple Agent',
      'Orange Agent'
    ];
    
    // Weapons and rooms remain the same
    this.WEAPONS = ['Candlestick', 'Dagger', 'Lead Pipe', 'Revolver', 'Rope', 'Wrench'];
    this.ROOMS = ['Kitchen', 'Ballroom', 'Conservatory', 'Dining Room', 'Billiard Room', 'Library', 'Lounge', 'Hall', 'Study'];
    
    // Game state
    this.solution = null;          // The murder solution (suspect, weapon, room)
    this.agents = [];              // List of agent instances
    this.currentTurn = 0;          // Current turn counter
    this.activeAgentIndex = 0;     // Index of the currently active agent
    this.gameLog = [];             // Complete log of game events
    this.activePlayers = 6;        // Number of active (not eliminated) players
    this.winner = null;            // The winning agent (if any)
    this.accusations = [];         // Track all accusations made during the game
    this.mode = mode;              // Game mode (spectate or play)
    this.humanPlayer = null;       // Human player instance (if any)
    this.modelsUsed = {};          // Map of models used by agents
    this.isOver = false;           // Flag indicating if the game is over

    // Advanced tracking
    this.turnHistory = [];         // History of turns for memory and analysis
    this.maxTurns = 120;           // Prevent infinite games
    this.models = [...MODEL_LIST].sort(() => Math.random() - 0.5); // Randomize model order
  }

  static setIO(io) {
    ioInstance = io;
  }

  logEvent(event) {
    const agent = event.agent ? this.agents.find(a => a.name === event.agent) : null;
    const modelInfo = agent ? ` (${agent.model})` : '';
    
    // Add timestamp if not present
    event.timestamp = event.timestamp || new Date();
    
    this.gameLog.push({
      ...event,
      model: agent?.model,
      timestamp: event.timestamp
    });

    // Use the message from the event if provided, otherwise format it
    const logMessage = event.message || this.formatEventMessage(event);

    // Print formatted log
    console.log(
      `[${event.type}] ${event.timestamp.toISOString()}\n` +
      `Agent: ${event.agent || 'System'}${modelInfo}\n` +
      `Details: ${logMessage}\n`
    );

    // Emit using the server's io instance
    if (this.io) {
      this.io.emit('game-event', {
        ...event,
        message: logMessage,
        timestamp: event.timestamp,
        agents: this.agents.map(a => ({
          name: a.name,
          hasLost: a.hasLost,
          model: a.model
        }))
      });
    }
  }

  createSolution() {
    // Just create and return the solution, don't set it
    const solution = {
      suspect: this.getRandomElement(this.SUSPECTS),
      weapon: this.getRandomElement(this.WEAPONS),
      room: this.getRandomElement(this.ROOMS)
    };

    return solution;  // Return solution without setting this.solution
  }

  /**
   * Gets all cards that are not part of the solution.
   * This method properly excludes solution cards from the deck.
   * @returns {Array} Array of card names that are not in the solution
   */
  getRemainingCards() {
    // Create a new array with all cards
    const allCards = [
      ...this.SUSPECTS.filter(card => card !== this.solution.suspect),
      ...this.WEAPONS.filter(card => card !== this.solution.weapon),
      ...this.ROOMS.filter(card => card !== this.solution.room)
    ];
    
    return allCards;
  }

  /**
   * Distributes the remaining cards fairly among the six agents.
   * @param {Array} cards - The remaining cards after solution selection
   * @returns {Array} Array of 6 card arrays, one for each agent
   */
  distributeCards(cards) {
    // Shuffle the cards first
    const shuffled = this.shuffle([...cards]);  // Make a copy to avoid modifying the original
    const dealtCards = [[], [], [], [], [], []]; // Create 6 empty arrays for agents
    
    // Distribute cards one at a time to ensure even distribution
    for (let i = 0; i < shuffled.length; i++) {
      const agentIndex = i % 6;  // Cycle through agents 0-5
      dealtCards[agentIndex].push(shuffled[i]);
    }
    
    // Debug: log card distribution to verify no duplicates
    logger.debug(`Card distribution validation: ${dealtCards.map(hand => hand.length).join(', ')} cards per agent`);
    
    return dealtCards;
  }

  getRandomElement(array) {
    return array[Math.floor(Math.random() * array.length)];
  }

  /**
   * Shuffles an array in place using the Fisher-Yates (Knuth) algorithm.
   * @param {Array} array - The array to shuffle.
   */
  shuffle(array) {
    let currentIndex = array.length,  randomIndex;
  
    // While there remain elements to shuffle.
    while (currentIndex > 0) {
  
      // Pick a remaining element.
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }
  
    return array;
  }

  // Add this method to get current game state
  getGameState() {
    return {
      currentTurn: this.currentTurn,
      activeAgent: this.agents[this.activeAgentIndex].name,
      availableSuspects: this.SUSPECTS,
      availableWeapons: this.WEAPONS,
      availableRooms: this.ROOMS,
      activePlayers: this.agents.filter(a => !a.hasLost).length,
      recentHistory: this.turnHistory.slice(-5)
    };
  }

  /**
   * Processes a complete turn for the current active agent.
   * 
   * The turn sequence is:
   * 1. Agent makes a suggestion (suspect, weapon, room)
   * 2. Other agents challenge if they can disprove the suggestion
   * 3. Agent considers making an accusation based on information gathered
   * 4. All agents update their memories with the turn's events
   * 
   * @async
   * @returns {Promise<Object>} Turn result containing suggestion and accusation decision
   */
  async processTurn() {
    if (this.isOver) return;
    const currentAgent = this.agents[this.activeAgentIndex];
    if (currentAgent.hasLost) {
        this.logger.log('info', `${currentAgent.name} has lost and is skipped.`);
        await this.nextTurn();
        return;
    }
    this.currentTurn++;
    this.logger.log('info', `Turn ${this.currentTurn}: ${currentAgent.name}'s turn.`);
    try {
        // Log agent's current memory and deductions for debugging
        this.logger.log('debug', `${currentAgent.name} Memory: ${currentAgent.memory.currentMemory || '(No memory)'}`);
        this.logger.log('debug', `${currentAgent.name} Known Cards: ${JSON.stringify(currentAgent.memory.knownCards)}`);
        this.logger.log('debug', `${currentAgent.name} Eliminated Cards: ${JSON.stringify(currentAgent.memory.eliminatedCards)}`);

        // 0. Move the agent (placeholder for now)
        console.log(`${currentAgent.name} is moving...`);
        currentAgent.move(this.ROOMS); // Call the agent's move method
        console.log(`${currentAgent.name} is now in room: ${currentAgent.location}`);

        // 1. Make suggestion (must be in the agent's current room)
        console.log(`${currentAgent.name} is making a suggestion...`);
        const suggestion = await currentAgent.makeSuggestion(this.getGameState());
        const turnResult = {
            agent: currentAgent.name,
            suggestion: suggestion,
            accusationDecision: null,
            theory: null
        };
        
        if (suggestion) {
            console.log(`Suggestion from ${currentAgent.name}:`, suggestion);
            this.emit('suggestion', {
                agent: currentAgent.name,
                suggestion: suggestion,
                reasoning: suggestion.reasoning || 'No reasoning provided',
                timestamp: new Date()
            });
        }

        // 2. Process challenges
        console.log('Processing challenges...');
        const challengeResult = await this.processChallenges(currentAgent, suggestion);
        
        if (challengeResult.canChallenge) {
            console.log(`Challenge successful by ${challengeResult.challengingAgent}`);
            this.emit('challenge', {
                agent: challengeResult.challengingAgent,
                cardShown: challengeResult.cardToShow,
                timestamp: new Date()
            });
        } else {
            console.log('No successful challenges');
        }

        // 3. After seeing challenge results, consider making an accusation
        console.log(`${currentAgent.name} is considering an accusation...`);
        logger.debug(`Calling agent.considerAccusation() for ${currentAgent.name}...`);
        
        let accusationDecision;
        try {
            accusationDecision = await currentAgent.considerAccusation();
            logger.debug(`agent.considerAccusation() completed for ${currentAgent.name}.`);
        } catch (error) {
            console.error(`Error during accusation consideration for ${currentAgent.name}:`, error.message);
            // If accusation fails, use a safe default (don't accuse)
            accusationDecision = {
                shouldAccuse: false,
                accusation: { suspect: null, weapon: null, room: null },
                confidence: { suspect: 0, weapon: 0, room: 0 },
                reasoning: "Error in accusation evaluation, choosing not to accuse."
            };
        }
        
        turnResult.accusationDecision = accusationDecision;
        console.log('Accusation decision:', accusationDecision);
        
        // 4. If agent decides to make accusation, process it
        if (accusationDecision.shouldAccuse) {
            console.log(`${currentAgent.name} is making an accusation!`);
            this.emit('accusation', {
                agent: currentAgent.name,
                accusation: accusationDecision.accusation,
                reasoning: accusationDecision.reasoning,
                confidence: accusationDecision.confidence,
                timestamp: new Date()
            });

            const isCorrect = this.checkAccusation(
                currentAgent,
                accusationDecision.accusation.suspect,
                accusationDecision.accusation.weapon,
                accusationDecision.accusation.room
            );

            if (!isCorrect) {
                currentAgent.setLost();
                this.logEvent({
                    type: 'loss',
                    agent: currentAgent.name,
                    message: `${currentAgent.name} made an incorrect accusation and was eliminated.`,
                    timestamp: new Date()
                });
                this.activePlayers--;
                
                // Check if only one player remains
                if (this.activePlayers === 1) {
                    const winner = this.agents.find(a => !a.hasLost);
                    if (winner) {
                        this.winner = winner;
                        this.handleGameOver('ONE_PLAYER_REMAINING', winner);
                    }
                }
            } else {
                // Correct accusation - game over
                this.winner = currentAgent;
                this.handleGameOver('CORRECT_ACCUSATION', currentAgent);
            }
        }

        // After dealing with the suggestion and challenges, update agent memories
        if (!this.isOver) {
            // Use our new function instead of the inline memory update code
            await this.updateAgentMemories(currentAgent, suggestion, challengeResult);
        }

        // Advance to next turn if game is not over
        if (!this.isOver) {
            this.nextTurn();
        }

        return turnResult;
    } catch (error) {
        console.error('Turn processing error:', error);
        
        // Try to advance to next turn despite error
        if (!this.isOver) {
            this.nextTurn();
        }
        
        return {
            agent: currentAgent.name,
            suggestion: null,
            accusationDecision: null,
            theory: null
        };
    }
  }
  
  /**
   * Calculate reward based on deduction accuracy.
   * 
   * @param {Array<string>} agentDeductions - Deductions made by the agent
   * @param {Array<string>} groundTruthDeductions - Ground truth deductions that should have been made
   * @returns {number} Reward value (0.0, 0.5, or 1.0)
   */
  calculateDeductionReward(agentDeductions, groundTruthDeductions) {
    // Handle edge cases
    if (!Array.isArray(agentDeductions)) {
      agentDeductions = [];
    }
    
    if (!Array.isArray(groundTruthDeductions) || groundTruthDeductions.length === 0) {
      // If there's nothing to deduce but agent claims deductions, penalize
      return agentDeductions.length > 0 ? 0.0 : 1.0;
    }
    
    // Convert to Sets for easier comparison
    const agentSet = new Set(agentDeductions);
    const truthSet = new Set(groundTruthDeductions);
    
    // Find correct and incorrect deductions
    let correctCount = 0;
    let incorrectExists = false;
    
    // Check each agent deduction against ground truth
    agentDeductions.forEach(deduction => {
      if (truthSet.has(deduction)) {
        correctCount++;
      } else {
        incorrectExists = true;
      }
    });
    
    // Calculate accuracy if there were any ground truth deductions
    if (incorrectExists) {
      return 0.0; // Any incorrect deduction results in zero reward
    }
    
    const accuracy = truthSet.size > 0 ? correctCount / truthSet.size : 1.0;
    
    // Assign reward based on accuracy thresholds
    if (accuracy >= 0.75) {
      return 1.0; // High accuracy
    } else if (accuracy >= 0.25) {
      return 0.5; // Medium accuracy
    } else {
      return 0.0; // Low accuracy
    }
  }

  async handleAccusation(agent, accusation) {
    const isCorrect = this.solution.suspect === accusation.suspect &&
                     this.solution.weapon === accusation.weapon &&
                     this.solution.room === accusation.room;

    // Emit accusation result
    this.emit('accusation', {
      agent: agent.name,
      accusation: accusation,
      result: isCorrect,
      timestamp: new Date()
    });

    if (isCorrect) {
      this.winner = agent.name;
      await this.endGame(agent);
    } else {
      this.eliminatePlayer(agent);
    }
  }

  async endGame() {
    if (this.isOver) return; // Prevent multiple calls
    
    this.isOver = true;
    console.log('Game ending condition met');
    
    try {
      // Save results before emitting game-over
      await this.saveGameResults();
      
      if (this.io) {
        this.io.emit('game-over', {
          winner: this.winner ? {
            name: this.winner.name,
            model: this.winner.model
          } : null,
          solution: this.solution
        });
      }

      // Log winner in a cleaner way
      if (this.winner) {
        console.log(`Game ended. Winner: ${this.winner.name} (${this.winner.model})`);
      } else {
        console.log('Game ended with no winner');
      }
    } catch (error) {
      console.error('Error in endGame:', error);
    }
  }

  isGameOver() {
    return this.isOver || this.winner !== null || this.activePlayers <= 1;
  }

  getNextActiveAgent() {
    let nextIndex = (this.activeAgentIndex + 1) % 6;
    while (this.agents[nextIndex].hasLost) {
      nextIndex = (nextIndex + 1) % 6;
    }
    return nextIndex;
  }

  /**
   * Processes challenges to a suggestion, following Cluedo turn order.
   *
   * Iterates through players clockwise starting from the suggester's left.
   * The first player who can disprove the suggestion does so by showing ONE card.
   * The process stops after the first successful challenge.
   *
   * @param {Agent} suggestingAgent - The agent who made the suggestion.
   * @param {Object} suggestion - The suggestion object {suspect, weapon, room}.
   * @returns {Promise<Object>} Challenge result containing:
   *                            - canChallenge: Boolean (true if any agent challenged)
   *                            - challengingAgent: Name of the agent who challenged (if any)
   *                            - cardToShow: The specific card shown (if any)
   */
  async processChallenges(suggestingAgent, suggestion) {
    const numAgents = this.agents.length;
    const startIndex = this.agents.findIndex(a => a.name === suggestingAgent.name);
    
    // Iterate clockwise starting from the agent to the left
    for (let i = 1; i < numAgents; i++) {
      const challengerIndex = (startIndex + i) % numAgents;
      const potentialChallenger = this.agents[challengerIndex];
      
      // Skip suggesting agent and eliminated players
      if (potentialChallenger.name === suggestingAgent.name || potentialChallenger.hasLost) {
        continue;
      }
      
      // Find which cards the potential challenger holds that match the suggestion
      const matchingCards = [
        suggestion.suspect,
        suggestion.weapon,
        suggestion.room
      ].filter(card => potentialChallenger.cards.has(card));
      
      // If the agent has matching cards, they MUST challenge
      if (matchingCards.length > 0) {
        logger.info(`${potentialChallenger.name} can challenge with card(s): ${matchingCards.join(', ')}`);
        
        // Let the agent decide which card to show (if multiple options)
        // The evaluateChallenge method should return only ONE card
        const challengeDecision = await potentialChallenger.evaluateChallenge(suggestion, matchingCards);
        
        if (challengeDecision && challengeDecision.cardToShow) {
          // Log the chosen card
          logger.info(`${potentialChallenger.name} chose to show the card: ${challengeDecision.cardToShow}`);
          
          // FIRST successful challenge stops the process
          return {
            canChallenge: true,
            challengingAgent: potentialChallenger.name,
            cardToShow: challengeDecision.cardToShow
          };
        } else {
          // This case should ideally not happen if matchingCards > 0, but handle defensively
          logger.warn(`${potentialChallenger.name} had matching cards but evaluateChallenge didn't return a cardToShow.`);
          // Continue to the next player just in case evaluateChallenge had an internal error
        }
      } else {
        logger.info(`${potentialChallenger.name} cannot challenge.`);
      }
    }
    
    // If loop completes, no one could challenge
    logger.info('No agent could challenge the suggestion.');
    return {
      canChallenge: false,
      challengingAgent: null,
      cardToShow: null
    };
  }

  formatNarrativeSuggestion(agentName, suggestion) {
    return `${agentName} suggests that the crime was committed by ${suggestion.suspect} in the ${suggestion.room} with the ${suggestion.weapon}.`;
  }

  formatNarrativeChallenge(agentName, card) {
    return `${agentName} shows a card to disprove the suggestion.`;
  }

  async updateAgentMemories(agent, suggestion, challengeResult) {
    const turnEvents = this.formatTurnEvents(agent, suggestion, challengeResult, agent);
    // Update memory for the agent who made the suggestion
    try {
        logger.info(`Updating memory for ${agent.name}...`);
        // Agent.updateMemory now returns { deducedCards, summary, error }
        const memoryUpdateResult = await agent.updateMemory(turnEvents);

        // If the update call itself had an error, log it but don't log trajectory
        if (memoryUpdateResult.error) {
            logger.error(`Memory update for ${agent.name} failed within Agent/LLMService: ${memoryUpdateResult.error}`);
            // Skip trajectory logging for this failed update
        } else {
            // Log deterministic deductions and calculate reward
            const groundTruthDeductions = this.getDeterministicNewDeductions(agent, turnEvents); 
            const reward = this.calculateDeductionReward(
                memoryUpdateResult.deducedCards || [], // Use cards deduced by LLM
                groundTruthDeductions // Compare against ground truth
            );

            // Get the request ID stored by Agent.js during the updateMemory call
            const requestId = agent.lastMemoryUpdateRequestId;

            this.logEvent({
                type: 'deduction_comparison', 
                agent: agent.name,
                llmDeductions: memoryUpdateResult.deducedCards || [],
                groundTruthDeductions: groundTruthDeductions,
                reward: reward,
                message: `LLM deduced ${memoryUpdateResult.deducedCards?.length || 0} cards. Ground truth: ${groundTruthDeductions.length}. Reward: ${reward}.`
            });

            // --- Log Trajectory to ART Wrapper (Conditional) ---
            if (process.env.LLM_BACKEND === 'ART') {
                if (requestId && typeof reward === 'number') {
                    const trajectoryPayload = {
                        request_id: requestId,
                        reward: reward,
                        metrics: { 
                            llm_deductions_count: memoryUpdateResult.deducedCards?.length || 0,
                            ground_truth_deductions_count: groundTruthDeductions.length
                            // Add any other relevant metrics
                        }
                    };
                    try {
                        logger.info(`Logging memory_update trajectory for ${agent.name}, request ${requestId}, reward ${reward}`);
                        await axios.post(`${ART_WRAPPER_URL}/log_trajectory`, trajectoryPayload);
                        // Optionally clear the request ID on the agent after logging?
                        // agent.lastMemoryUpdateRequestId = null; 
                    } catch (logError) {
                        logger.error(`Failed to log trajectory for request ${requestId}: ${logError.message}`, { error: logError });
                        // Handle logging failure if needed (e.g., retry?)
                    }
                } else {
                    if (!requestId) logger.warn(`[ART Mode] Cannot log memory_update trajectory for ${agent.name}: Missing request ID.`);
                    if (typeof reward !== 'number') logger.warn(`[ART Mode] Cannot log memory_update trajectory for ${agent.name}: Invalid reward (${reward}).`);
                }
            }
            // ----------------------------------------
        }

    } catch (error) {
        logger.error(`Error processing memory update for ${agent.name}: ${error.message}`, { error });
    }

    // Update memory for other agents based on observable events
    for (const otherAgent of this.agents) {
        if (otherAgent !== agent && !otherAgent.hasLost) {
            const observableEvents = this.formatTurnEvents(agent, suggestion, challengeResult, otherAgent);
            try {
                logger.info(`Updating memory for observer ${otherAgent.name}...`);
                // Call updateMemory, but we generally don't reward observers directly for passive updates
                const observerUpdateResult = await otherAgent.updateMemory(observableEvents); 
                if (observerUpdateResult.error){
                    logger.error(`Observer memory update for ${otherAgent.name} failed: ${observerUpdateResult.error}`);
                } 
                // --- Log Observer Trajectory (Optional & Conditional) ---
                if (process.env.LLM_BACKEND === 'ART') {
                    const observerRequestId = otherAgent.lastMemoryUpdateRequestId;
                    if (observerRequestId) {
                        const observerPayload = {
                            request_id: observerRequestId,
                            reward: 0, // Or calculate a specific observer reward
                            metrics: { 
                               is_observer: true, 
                               llm_deductions_count: observerUpdateResult.deducedCards?.length || 0 
                            }
                        };
                        try {
                            logger.info(`Logging observer memory_update trajectory for ${otherAgent.name}, request ${observerRequestId}, reward 0`);
                            await axios.post(`${ART_WRAPPER_URL}/log_trajectory`, observerPayload);
                            // otherAgent.lastMemoryUpdateRequestId = null; 
                        } catch (logError) {
                             logger.error(`Failed to log observer trajectory for request ${observerRequestId}: ${logError.message}`, { error: logError });
                        }
                    } else {
                         // No request ID likely means the update was skipped (no events) or failed before LLM call
                         if (observableEvents && observableEvents.length > 0 && !observerUpdateResult.error){
                             logger.warn(`[ART Mode] Cannot log observer memory_update trajectory for ${otherAgent.name}: Missing request ID.`);
                         }
                    }
                }
                // ----------------------------------------
            } catch (error) {
                logger.error(`Error processing observer memory update for ${otherAgent.name}: ${error.message}`, { error });
            }
        }
    }
  }

  getGameSummary() {
    return {
      currentTurn: this.currentTurn,
      activeAgent: this.agents[this.activeAgentIndex].name,
      agents: this.agents.map(agent => ({
        name: agent.name,
        isAlive: !agent.hasLost,
        model: agent.model,
        cards: this.mode === 'spectate' ? Array.from(agent.cards) : undefined
      })),
      winner: this.winner,
      solution: this.solution,
      gameLog: this.gameLog,
      turnHistory: this.turnHistory
    };
  }

  /**
   * Initializes a new game with random solution, distributed cards, and agents.
   * 
   * This method:
   * 1. Selects a random solution (murderer, weapon, location)
   * 2. Distributes remaining cards among agents
   * 3. Creates agent instances and assigns them LLM models
   * 4. Resets the game state
   * 
   * @async
   * @returns {Promise<boolean>} True if initialization was successful
   * @throws {Error} If initialization fails
   */
  async initialize() {
    try {
      // Reset memories
      if (this.agents) {
        this.agents.forEach(agent => agent.memory.reset());
      }

      // Start with fresh arrays for card distribution
      const suspects = [...this.SUSPECTS]; // Make copies to avoid modifying the originals
      const weapons = [...this.WEAPONS];
      const rooms = [...this.ROOMS];

      // Select the solution (by removing from the arrays)
      this.solution = {
        suspect: suspects.splice(Math.floor(Math.random() * suspects.length), 1)[0],
        weapon: weapons.splice(Math.floor(Math.random() * weapons.length), 1)[0],
        room: rooms.splice(Math.floor(Math.random() * rooms.length), 1)[0]
      };
      
      // Log solution with a clear, consistent format
      logger.info(`*** CORRECT SOLUTION: Suspect=${this.solution.suspect}, Weapon=${this.solution.weapon}, Room=${this.solution.room} ***`);
      
      // Emit solution to clients if needed
      if (this.io) {
        this.io.emit('game-solution', this.solution);
      }

      // Combine the remaining cards (which now EXCLUDE the solution cards)
      const remainingCards = [...suspects, ...weapons, ...rooms];
      
      // Shuffle and distribute the remaining cards
      const dealtCards = this.distributeCards(remainingCards);

      // Initialize agents
      this.agents = this.AGENT_NAMES.map((name, index) => {
        const model = this.models[index % this.models.length];
        const agent = new Agent(name, dealtCards[index], model, this);
        agent.game = this; // Ensure game reference is set
        return agent;
      });

      // Initialize game state
      this.currentTurn = 0;
      this.activeAgentIndex = 0;
      this.gameLog = [];
      this.turnHistory = [];
      this.isOver = false;
      this.startTime = Date.now();

      // Validate the game setup is correct
      const setupValid = this.validateGameSetup();
      if (!setupValid) {
        logger.error("Game setup failed validation - solution cards may be duplicated in agent hands!");
      }

      // Log initialization
      this.logEvent({
        type: 'GAME_INIT',
        timestamp: new Date(),
        message: `Game initialized with ${this.agents.length} agents`
      });

      return true;
    } catch (error) {
      console.error('Game initialization failed:', error);
      throw error;
    }
  }

  eliminatePlayer(agent) {
    agent.setLost();
    this.activePlayers--;
    
    // Check if game should end (only one player left)
    if (this.activePlayers === 1) {
      const winner = this.agents.find(a => !a.hasLost);
      this.endGame(winner);
    }
  }

  async processHumanAction(action) {
    const activeAgent = this.agents[this.activeAgentIndex];
    
    if (!(activeAgent instanceof HumanAgent)) {
      console.error('Received human action but it\'s not human player\'s turn');
      return;
    }
    
    // Update position for suggestions
    if (action.type === 'suggestion') {
      activeAgent.position = action.room;
    }
    
    if (action.type === 'accusation') {
      const accusation = {
        suspect: action.suspect,
        weapon: action.weapon,
        room: action.room
      };
      
      const isCorrect = 
        accusation.suspect === this.solution.suspect &&
        accusation.weapon === this.solution.weapon &&
        accusation.room === this.solution.room;
      
      if (isCorrect) {
        this.winner = activeAgent.name;
        this.logEvent({
          type: 'GAME_OVER',
          agent: activeAgent.name,
          message: `${activeAgent.name} won with a correct accusation!`,
          timestamp: new Date()
        });
      } else {
        this.eliminatePlayer(activeAgent);
        this.logEvent({
          type: 'ACCUSATION_FAILED',
          agent: activeAgent.name,
          message: `${activeAgent.name} made an incorrect accusation and was eliminated.`,
          timestamp: new Date()
        });
      }
    }
    
    if (action.type === 'suggestion') {
      // Process suggestion
      const suggestion = {
        suspect: action.suspect,
        weapon: action.weapon,
        room: action.room
      };
      
      const narrativeSuggestion = this.formatNarrativeSuggestion(
        activeAgent.name,
        suggestion
      );
      
      this.logEvent({
        type: 'SUGGESTION',
        agent: activeAgent.name,
        suggestion,
        narrativeText: narrativeSuggestion,
        timestamp: new Date()
      });
      
      // Process challenges
      const challengeResult = await this.processChallenges(activeAgent, suggestion);
      
      // Update memories
      await this.updateAgentMemories({
        type: 'TURN_COMPLETE',
        turnNumber: this.currentTurn,
        activeAgent: activeAgent.name,
        suggestion,
        challengeResult,
        narrativeSuggestion,
        timestamp: new Date()
      });
    }
    
    // Move to next turn
    this.currentTurn++;
    this.activeAgentIndex = (this.activeAgentIndex + 1) % 6;
    
    // Process next turn immediately
    await this.processTurn();

    // Emit updated game state
    if (this.io) {
      this.io.emit('game-state', this.getGameSummary());
    }
    
    // Return true to indicate the action was processed
    return true;
  }

  // Check valid moves between rooms
  isValidMove(fromRoom, toRoom) {
    const roomConnections = {
      'Study': ['Hall', 'Library'],
      'Hall': ['Study', 'Lounge', 'Billiard Room'],
      'Lounge': ['Hall', 'Dining Room'],
      'Dining Room': ['Lounge', 'Kitchen', 'Billiard Room'],
      'Kitchen': ['Dining Room', 'Ballroom'],
      'Ballroom': ['Kitchen', 'Conservatory', 'Billiard Room'],
      'Conservatory': ['Ballroom', 'Library'],
      'Library': ['Conservatory', 'Study', 'Billiard Room'],
      'Billiard Room': ['Library', 'Ballroom', 'Dining Room', 'Hall']
    };

    return roomConnections[fromRoom]?.includes(toRoom) || false;
  }

  processAccusation(accusingPlayer, accusation) {
    const isCorrect = this.solution.suspect === accusation.suspect &&
                     this.solution.weapon === accusation.weapon &&
                     this.solution.room === accusation.room;

    if (isCorrect) {
      this.endGame(accusingPlayer);
    } else {
      this.eliminatePlayer(accusingPlayer);
      this.checkGameEnd();
    }
  }

  checkGameEnd() {
    if (this.activePlayers === 0) {
      this.endGame(null); // No winner
    }
  }

  checkGameCompletion() {
    const aliveAgents = this.agents.filter(agent => agent.isAlive);
    // Game should end when only 1 agent remains
    return aliveAgents.length <= 1;
  }

  async nextPhase() {
    // Original phase transition logic
    if (this.phase === 'night') {
      this.phase = 'day';
    } else {
      this.phase = 'night';
    }
    
    // Should add immediate completion check after phase change
    if (this.checkGameCompletion()) {
      await this.endGame();
    }
  }

  // Add this method to the Game class
  nextTurn() {
    // Increment turn counter
    this.currentTurn++;
    
    // Move to next active agent
    let nextIndex = (this.activeAgentIndex + 1) % this.agents.length;
    
    // Skip eliminated players
    while (this.agents[nextIndex].hasLost && nextIndex !== this.activeAgentIndex) {
      nextIndex = (nextIndex + 1) % this.agents.length;
    }
    
    this.activeAgentIndex = nextIndex;

    // Update turn history
    this.turnHistory.push({
      turnNumber: this.currentTurn,
      activeAgent: this.agents[this.activeAgentIndex].name
    });

    // Check for max turns reached
    if (this.currentTurn >= this.maxTurns) {
      console.log('Game ended due to maximum turns reached');
      this.endGame();
    }
  }

  // Add this method to the Game class
  checkAccusation(agent, suspect, weapon, room) {
    const isCorrect =
      suspect === this.solution.suspect &&
      weapon === this.solution.weapon &&
      room === this.solution.room;

    this.logEvent({
      type: 'accusationResult',
      agent: agent.name,
      accusation: { suspect, weapon, room },
      isCorrect: isCorrect,
      timestamp: new Date()
    });

    if (isCorrect) {
      this.logger.log('info', `${agent.name} made a CORRECT accusation! ${suspect}, ${weapon}, ${room}`);
      this.winner = agent;
      this.handleGameOver('CORRECT_ACCUSATION', agent); // End game on correct accusation
    } else {
      this.logger.log('info', `${agent.name} made an INCORRECT accusation.`);
      agent.setLost(); // Mark the agent as lost
      this.logEvent({
        type: 'loss',
        agent: agent.name,
        message: `${agent.name} made an incorrect accusation and was eliminated.`,
        timestamp: new Date()
      });

      // Check if this elimination ends the game (only one player left)
      const remainingPlayers = this.getRemainingPlayers();
      if (remainingPlayers.length === 1) {
          this.logger.log('info', `Only one player (${remainingPlayers[0].name}) remains after incorrect accusation.`);
          this.winner = remainingPlayers[0]; // The last remaining player wins
          this.handleGameOver('LAST_PLAYER_STANDING', this.winner);
      } else if (remainingPlayers.length === 0) {
        // Edge case: Should not happen if game starts with >1 player, but handle defensively
         this.logger.log('warn', `No players remaining after incorrect accusation.`);
         this.handleGameOver('NO_WINNER', null);
      }
      // If > 1 player remains, the game continues automatically via nextTurn
    }
    return isCorrect; // Return the result of the check
  }

  async saveGameResults() {
    try {
      const __dirname = path.dirname(fileURLToPath(import.meta.url));
      const resultsPath = path.join(__dirname, '../../game_results.json');
      
      // Read existing results or create new array
      let existingResults = [];
      
      // Create directory if it doesn't exist
      const dir = path.dirname(resultsPath);
      if (!fs.existsSync(dir)) {
        await fs.promises.mkdir(dir, { recursive: true });
      }

      // Try to read existing file, start with empty array if file doesn't exist
      if (fs.existsSync(resultsPath)) {
        try {
          const fileContent = await fs.promises.readFile(resultsPath, 'utf8');
          existingResults = JSON.parse(fileContent);
        } catch (error) {
          console.error('Error reading existing results, starting fresh:', error);
          // Continue with empty array if file is invalid
        }
      }

      // Create new game result using the enhanced GameResult class
      const gameResultObj = new GameResult(this);
      const gameResult = gameResultObj.toJSON();

      // Append new result
      existingResults.push(gameResult);

      // Write/create file
      try {
        await fs.promises.writeFile(
          resultsPath, 
          JSON.stringify(existingResults, null, 2),
          { encoding: 'utf8' }
        );
        console.log('Game results saved successfully to:', resultsPath);
      } catch (writeError) {
        console.error('Error writing game results:', writeError);
        throw writeError;
      }
    } catch (error) {
      console.error('Error in saveGameResults:', error);
      throw error;
    }
  }

  // Add helper method for formatting event messages
  formatEventMessage(event) {
    switch (event.type) {
      case 'SOLUTION':
        return `Solution: ${event.solution.suspect} in the ${event.solution.room} with the ${event.solution.weapon}`;
      case 'GAME_INIT':
        return `Game initialized with ${this.mode === 'play' ? 'human player' : '6 AI agents'}`;
      case 'SUGGESTION':
        return `${event.agent} suggests that the crime was committed by ${event.suggestion.suspect} in the ${event.suggestion.room} with the ${event.suggestion.weapon}`;
      case 'CHALLENGE':
        return `${event.agent} showed ${event.cardShown} to disprove the suggestion`;
      case 'SUGGESTION_REASONING':
        return event.suggestion.reasoning;
      case 'ACCUSATION_FAILED':
        return `${event.agent} made an incorrect accusation and was eliminated.`;
      case 'GAME_OVER':
        return `${event.agent} won with a correct accusation!`;
      default:
        return JSON.stringify(event);
    }
  }

  /**
   * Determines which cards can be definitively deduced as held by players based on a turn's events.
   * This creates the "ground truth" for deduction accuracy evaluation.
   * 
   * @param {Agent} agent - The agent for whom to calculate deductions
   * @param {Object} turnEvents - Events that occurred during the turn
   * @returns {Array<string>} Array of card names that can now be deterministically deduced
   */
  getDeterministicNewDeductions(agent, turnEvents) {
    // Cards that can be deterministically deduced from this turn's events
    const newDeductions = new Set();
    
    // Skip if no turn events
    if (!turnEvents || !Array.isArray(turnEvents) || turnEvents.length === 0) {
      return [];
    }

    // Get the agent's current known cards (that we'll exclude from "new" deductions)
    const agentKnownCards = new Set(agent.cards); // Start with agent's own cards
    
    // Extract suggestion information from turn events
    let suggestion = null;
    let suggester = null;
    let challengeResults = {};

    // First pass to extract suggestion and challenges information
    for (const event of turnEvents) {
      // Handle structured event objects or string events
      const eventText = typeof event === 'string' ? event : event.message || '';
      
      // Extract suggestion details
      const suggestionMatch = eventText.match(/(\w+ Agent) suggested ([\w\s]+), ([\w\s]+), ([\w\s]+)/);
      if (suggestionMatch) {
        const [_, suggestingAgent, suspect, weapon, room] = suggestionMatch;
        suggester = suggestingAgent;
        suggestion = { suspect, weapon, room };
      }
      
      // Extract challenge results
      const challengeMatch = eventText.match(/(\w+ Agent) (showed|did not show) (?:a|the) card(?: "([\w\s]+)")? to (\w+ Agent)/);
      if (challengeMatch) {
        const [_, challengingAgent, result, cardShown, toAgent] = challengeMatch;
        challengeResults[challengingAgent] = {
          showed: result === 'showed',
          cardShown: cardShown || null,
          toAgent
        };
      }
      
      // Specifically handle events where the card shown is known
      if (typeof event === 'object' && event.type === 'cardShown' && event.to === agent.name) {
        if (!agentKnownCards.has(event.card)) {
          newDeductions.add(event.card);
        }
      }
    }
    
    // If we have a suggestion, process deterministic deductions
    if (suggestion && suggester) {
      const suggestedCards = [suggestion.suspect, suggestion.weapon, suggestion.room];
      
      // Case 1: Agent made the suggestion and was shown a card
      if (suggester === agent.name) {
        // Find which agents showed cards
        const showingAgents = Object.entries(challengeResults)
          .filter(([_, result]) => result.showed && result.toAgent === agent.name)
          .map(([agentName]) => agentName);
          
        if (showingAgents.length === 1) {
          // If the agent knows all but one of the suggested cards, the shown card must be the missing one
          const heldSuggestionCards = suggestedCards.filter(card => agentKnownCards.has(card));
          
          if (heldSuggestionCards.length === 2) {
            // Agent must have been shown the third card
            const deducedCard = suggestedCards.find(card => !agentKnownCards.has(card));
            if (deducedCard) {
              newDeductions.add(deducedCard);
            }
          }
        }
      }
      
      // Case 2: Another agent made the suggestion and everyone responded "no" except one agent
      else {
        const noResponseAgents = Object.entries(challengeResults)
          .filter(([_, result]) => !result.showed)
          .map(([agentName]) => agentName);
          
        // Check if all but one agent (and the suggester) couldn't disprove
        const shouldHaveResponded = this.agents
          .filter(a => !a.hasLost && a.name !== suggester)
          .length;
          
        if (noResponseAgents.length === shouldHaveResponded - 1) {
          // The one agent who did show a card must have one of the suggested cards
          // If that agent showed a card to the current agent, we already handled it above
          // This deduction is more useful for the suggester, not the current agent
          
          // However, if all agents except one said "no" and that agent didn't show a card to the current agent,
          // we can deduce that the remaining agent must have at least one of the cards
          // (This is useful information but not a deterministic card deduction)
        }
        
        // Case 3: If all agents (except suggester) said "no", all three suggested cards must be in the solution
        // This is a strong deduction that the cards are NOT held by any player
        if (noResponseAgents.length === shouldHaveResponded) {
          // While this is deterministic knowledge, it's not a "card held by a player" deduction,
          // so we don't add it to newDeductions in this implementation
        }
      }
      
      // Case 4: If an agent couldn't disprove a suggestion, they must not have any of the suggested cards
      // We can deduce which agents definitely don't have which cards, but since our goal is to identify
      // cards that are definitely held by specific players, we don't add these negative deductions
    }
    
    // Convert the Set to an Array before returning
    return Array.from(newDeductions);
  }

  async handleGameOver(reason, agent) {
    // Implementation
  }
  
  /**
   * Formats turn events for agent memory updates
   * 
   * @param {Agent} suggestingAgent - The agent who made the suggestion
   * @param {Object} suggestion - The suggestion made
   * @param {Object} challengeResult - The challenge result
   * @param {Agent} recipientAgent - The agent who will receive these events
   * @returns {Array} Formatted turn events with appropriate information hiding
   */
  formatTurnEvents(suggestingAgent, suggestion, challengeResult, recipientAgent) {
    const events = [];
    
    // Add suggestion event - visible to all agents
    events.push({
      type: 'suggestion',
      agent: suggestingAgent.name,
      suspect: suggestion.suspect,
      weapon: suggestion.weapon,
      room: suggestion.room
    });
    
    // Add challenge event if there was a challenge
    if (challengeResult) {
      // Basic challenge info - visible to all agents
      events.push({
        type: 'challenge',
        agent: suggestingAgent.name, 
        challengingAgent: challengeResult.challengingAgent,
        canChallenge: challengeResult.canChallenge,
        // Don't include cardToShow here as it's private
      });
      
      // Add card shown event if a card was shown - BUT only show the specific card
      // to the agent who showed it or the agent who received it
      if (challengeResult.canChallenge && challengeResult.cardToShow) {
        // Determine if this recipient should know the specific card
        const shouldKnowSpecificCard = 
          recipientAgent.name === suggestingAgent.name || // the agent who received the card
          recipientAgent.name === challengeResult.challengingAgent; // the agent who showed the card
        
        if (shouldKnowSpecificCard) {
          // Full information for involved agents
          events.push({
            type: 'cardShown',
            from: challengeResult.challengingAgent,
            to: suggestingAgent.name,
            card: challengeResult.cardToShow
          });
        } else {
          // Limited information for other agents - they only know a card was shown, not which one
          events.push({
            type: 'cardShown',
            from: challengeResult.challengingAgent,
            to: suggestingAgent.name,
            card: null, // No specific card info
            hiddenInfo: true // Flag indicating info is hidden
          });
        }
      }
    }
    
    return events;
  }

  /**
   * Validates that the game setup is correct:
   * - Solution cards are not held by any agent
   * - No cards are duplicated across agent hands
   * This helps catch logic errors in card distribution.
   */
  validateGameSetup() {
    // Track which cards are held by all players
    const heldCards = new Set();
    
    // Check each agent's cards
    for (const agent of this.agents) {
      for (const card of agent.cards) {
        // Check for duplicate cards across agents
        if (heldCards.has(card)) {
          logger.error(`GAME SETUP ERROR: Card "${card}" is held by multiple agents`);
        }
        heldCards.add(card);
        
        // Check if any solution cards are incorrectly held by agents
        if (card === this.solution.suspect || 
            card === this.solution.weapon || 
            card === this.solution.room) {
          logger.error(`GAME SETUP ERROR: Solution card "${card}" is incorrectly held by ${agent.name}`);
          return false;
        }
      }
    }
    
    // Count total cards for additional validation
    const uniqueHeldCards = Array.from(heldCards);
    const expectedCardCount = 
      (this.SUSPECTS.length - 1) + 
      (this.WEAPONS.length - 1) + 
      (this.ROOMS.length - 1);
    
    if (uniqueHeldCards.length !== expectedCardCount) {
      logger.error(`GAME SETUP ERROR: Expected ${expectedCardCount} unique cards held by agents, but found ${uniqueHeldCards.length}`);
      return false;
    }
    
    logger.info(`Game setup validated: All ${uniqueHeldCards.length} cards correctly distributed, no solution cards in agent hands`);
    return true;
  }

  /**
   * Returns an array of agents who are still active in the game (have not lost).
   * @returns {Agent[]}
   */
  getRemainingPlayers() {
      return this.agents.filter(agent => !agent.hasLost);
  }
}