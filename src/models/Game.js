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
import { SUSPECTS, WEAPONS, ROOMS } from '../config/gameConstants.js';

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
    // Initialize instance properties from the exported constants
    this.SUSPECTS = [...SUSPECTS];
    this.WEAPONS = [...WEAPONS];
    this.ROOMS = [...ROOMS];

    // Distinct agent names (colors)
    this.AGENT_NAMES = [
      'Red Agent',
      'Blue Agent',
      'Green Agent',
      'Yellow Agent',
      'Purple Agent',
      'Orange Agent'
    ];
    
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
      activePlayers: this.agents.filter(a => !a.hasLost).length
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

        // --- FIX RESTORED: Update memories AFTER challenge, BEFORE accusation consideration ---
        if (!this.isOver) {
            await this.updateAgentMemories(currentAgent, suggestion, challengeResult, null);
        }
        // ----------------------------------------------------------------------

        // 3. After seeing challenge results AND UPDATED MEMORY, consider making an accusation
        console.log(`${currentAgent.name} is considering an accusation...`);
        logger.debug(`Calling agent.considerAccusation() for ${currentAgent.name}...`);
        
        let accusationDecision;
        try {
            // Pass current turn's suggestion and challenge result
            accusationDecision = await currentAgent.considerAccusation(suggestion, challengeResult);
            logger.debug(`agent.considerAccusation() completed for ${currentAgent.name}.`);
        } catch (error) {
            console.error(`Error during accusation consideration for ${currentAgent.name}:`, error.message);
            // If accusation fails, use a safe default (don't accuse)
            accusationDecision = {
                shouldAccuse: false,
                accusation: { suspect: null, weapon: null, room: null },
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

        // --- Check for missed deterministic accusation opportunities ---
        if (!this.isOver) {
            // Calculate if this agent could have made a deterministic accusation
            const allKnownEliminated = new Set([...currentAgent.cards, ...currentAgent.memory.eliminatedCards]); 
            const remainingSuspects = this.SUSPECTS.filter(s => !allKnownEliminated.has(s));
            const remainingWeapons = this.WEAPONS.filter(w => !allKnownEliminated.has(w));
            const remainingRooms = this.ROOMS.filter(r => !allKnownEliminated.has(r));

            if (remainingSuspects.length === 1 && remainingWeapons.length === 1 && remainingRooms.length === 1) {
                const deterministicAccusation = {
                    suspect: remainingSuspects[0],
                    weapon: remainingWeapons[0], 
                    room: remainingRooms[0],
                };

                // Check if this deterministic accusation matches the actual solution
                const isDeterministicallyCorrect = 
                    deterministicAccusation.suspect === this.solution.suspect &&
                    deterministicAccusation.weapon === this.solution.weapon &&
                    deterministicAccusation.room === this.solution.room;
                
                // Log if the agent could have accused correctly but didn't
                if (isDeterministicallyCorrect && (!accusationDecision || !accusationDecision.shouldAccuse)) {
                    logger.warn(`MISSED OPPORTUNITY: ${currentAgent.name} could have deterministically accused correctly ${JSON.stringify(deterministicAccusation)} but chose not to (Decision: ${JSON.stringify(accusationDecision)}).`);
                    await LoggingService.logLLMInteraction({
                        type: 'missed_deterministic_accusation',
                        agent: currentAgent.name,
                        model: currentAgent.model,
                        deterministicAccusation: deterministicAccusation,
                        llmDecision: accusationDecision,
                        timestamp: new Date()
                    });
                }
            }
        }
        // ----------------------------------------------------------------------

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

  /**
   * Updates agent memories based on the turn's events (suggestion, challenge).
   * Now calculates and logs deduction comparisons for ALL agents.
   *
   * @param {Agent} suggestingAgent - The agent who made the suggestion.
   * @param {Object} suggestion - The suggestion made {suspect, weapon, room, reasoning}.
   * @param {Object} challengeResult - Result of challenges {challenger, cardShown, noChallenge}.
   * @param {Object} accusationDecision - The accusation decision made by the suggesting agent this turn.
   * @async
   */
  async updateAgentMemories(suggestingAgent, suggestion, challengeResult, accusationDecision) {
      this.logger.info(`Updating memories for all agents after ${suggestingAgent.name}'s turn.`);
      this.logger.debug(`Suggestion details: ${JSON.stringify(suggestion)}`);
      this.logger.debug(`Challenge result details: ${JSON.stringify(challengeResult)}`);

      // Loop through ALL agents to update memory and compare deductions
      for (const agent of this.agents) {
          if (agent.hasLost) {
              this.logger.debug(`Skipping memory update for eliminated agent: ${agent.name}`);
              continue;
          }

          this.logger.info(`Processing memory update for agent: ${agent.name}`);
          this.logger.debug(`Agent cards: ${Array.from(agent.cards).join(', ')}`);
          this.logger.debug(`Agent eliminated cards: ${Array.from(agent.memory.eliminatedCards).join(', ')}`);

          try {
              // --- Add Deterministic Memory Update based on Challenge --- 
              if (challengeResult && challengeResult.cardToShow) {
                  if (agent.name === suggestingAgent.name) {
                      // Card was shown TO the suggesting agent
                      if (!agent.cards.has(challengeResult.cardToShow)) { // Avoid eliminating own card if shown back
                          agent.memory.eliminateCard(challengeResult.cardToShow);
                          this.logger.debug(`Deterministically eliminated ${challengeResult.cardToShow} for ${agent.name} (was shown card).`);
                      }
                  } else if (agent.name === challengeResult.challengingAgent) {
                      // Challenger showed the card (should already be in knownCards, but ensure consistency)
                      // No action needed here as addKnownCard handles removing from other sets
                      // agent.memory.addKnownCard(challengeResult.cardToShow);
                  } else {
                      // Observer agent: if they weren't the suggester or challenger, the shown card is eliminated for them
                      agent.memory.eliminateCard(challengeResult.cardToShow);
                      this.logger.debug(`Deterministically eliminated ${challengeResult.cardToShow} for ${agent.name} (observed challenge).`);
                  }
              }
              // --------------------------------------------------------

              // --- Record Structured Turn Events in Memory History --- 
              // (Do this BEFORE calling the LLM for memory update)
              agent.memory.recordTurnEvents(
                  this.currentTurn, // Pass the current game turn number
                  suggestingAgent ? suggestingAgent.name : null, // Pass the suggester's name
                  suggestion, // Pass the structured suggestion object
                  challengeResult // Pass the structured challenge result object
              );
              this.logger.debug(`Recorded turn events in memory history for ${agent.name}.`);
              // -----------------------------------------------------

              // Calculate Ground Truth Deductions for THIS agent
              this.logger.debug(`Calculating ground truth deductions for ${agent.name}...`);
              const groundTruthDeductions = this.getDeterministicNewDeductions(agent, suggestingAgent, suggestion, challengeResult);
              this.logger.info(`Ground truth deductions for ${agent.name}: ${groundTruthDeductions.length > 0 ? groundTruthDeductions.join(', ') : 'None'}`);

              // Call LLM to update memory and get its deductions
              const turnEventsForLLM = this.formatTurnEvents(suggestingAgent, suggestion, challengeResult, agent);
              this.logger.debug(`Turn events for ${agent.name}: ${JSON.stringify(turnEventsForLLM)}`);
              
              const llmMemoryUpdateResult = await LLMService.updateMemory(agent, agent.memory, turnEventsForLLM);
              const llmDeductions = llmMemoryUpdateResult.deducedCards || [];
              this.logger.info(`LLM deductions for ${agent.name}: ${llmDeductions.length > 0 ? llmDeductions.join(', ') : 'None'}`);

              // Calculate Reward and Log Comparison
              const reward = this.calculateDeductionReward(llmDeductions, groundTruthDeductions);
              this.logger.info(`Reward calculation for ${agent.name}:
                - LLM deductions: ${llmDeductions.join(', ') || 'None'}
                - Ground truth: ${groundTruthDeductions.join(', ') || 'None'}
                - Reward: ${reward}`);

              // Log the comparison event
              await LoggingService.logLLMInteraction({
                type: 'deduction_comparison', 
                agent: agent.name,
                  model: agent.model,
                  llmDeductions: llmDeductions,
                groundTruthDeductions: groundTruthDeductions,
                reward: reward,
                  timestamp: new Date()
              });

          } catch (error) {
              this.logger.error(`Error updating memory for ${agent.name}: ${error.message}`, { error });
          }
      }

      this.logger.info('Finished updating memories for all agents.');
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
   * Calculates deterministic deductions an agent can make based on turn events
   * and their current knowledge (hand + eliminated cards).
   *
   * @param {Agent} agent - The agent for whom deductions are being calculated.
   * @param {Agent} suggestingAgent - The agent who made the suggestion.
   * @param {Object | null} suggestion - The suggestion made {suspect, weapon, room}, or null if none.
   * @param {Object} challengeResult - Result of challenges {challengingAgent, cardShown, noChallenge}.
   * @returns {Array<string>} - An array of card names newly deduced this turn.
   */
  getDeterministicNewDeductions(agent, suggestingAgent, suggestion, challengeResult) {
      this.logger.debug(`Starting deterministic deductions for ${agent.name}`);
      this.logger.debug(`Input state:
        - Current turn: ${this.currentTurn}
        - Suggesting agent: ${suggestingAgent.name}
        - Suggestion: ${JSON.stringify(suggestion)}
        - Challenge result: ${JSON.stringify(challengeResult)}
        - Agent cards: ${Array.from(agent.cards).join(', ')}
        - Agent eliminated cards: ${Array.from(agent.memory.eliminatedCards).join(', ')}`);

      // Initialize with ONLY previously eliminated cards
      const agentKnownCards = new Set([...agent.memory.eliminatedCards]);
      const newlyDeducedCards = new Set();

      // Rule 0: On first memory update, add own cards as newly deduced AND to the known set
      const isFirstUpdate = !agent.memory.currentMemory;
      if (isFirstUpdate) {
          this.logger.debug(`Rule 0 triggered for ${agent.name} (first update)`);
          for (const card of agent.cards) {
              // Always add own cards as deductions on the first update
              this.logger.debug(`Rule 0: Adding own card ${card} as newly deduced`);
              newlyDeducedCards.add(card);
              agentKnownCards.add(card); // Also add to known set for subsequent rules
          }
      } else {
          // If not the first update, add own cards to the 'known' set for checks, but NOT as 'newly deduced'
          for (const card of agent.cards) {
              agentKnownCards.add(card);
        }
      }

      if (!suggestion || !suggestion.suspect) {
          this.logger.debug(`No suggestion provided for ${agent.name}, returning deductions: ${Array.from(newlyDeducedCards).join(', ')}`);
          return Array.from(newlyDeducedCards);
      }

      // Rule 1: If this agent is the suggester and a card was shown to them, they can deduce it
      if (agent.name === suggestingAgent.name && !challengeResult.noChallenge && challengeResult.cardToShow) {
          const shownCard = challengeResult.cardToShow;
          if (!agentKnownCards.has(shownCard)) {
              this.logger.debug(`Rule 1: ${agent.name} was shown ${shownCard} and can deduce it`);
              newlyDeducedCards.add(shownCard);
              agentKnownCards.add(shownCard);
        }
      }
      
      // Rule 2: If this agent is the suggester and they hold 2 of the 3 suggested cards,
      // and someone challenged, they can deduce the third card
      if (agent.name === suggestingAgent.name && !challengeResult.noChallenge) {
          const suggestionCards = [suggestion.suspect, suggestion.weapon, suggestion.room];
          const cardsAgentHolds = suggestionCards.filter(card => agent.cards.has(card));
          
          if (cardsAgentHolds.length === 2) {
              const deducedCard = suggestionCards.find(card => !agent.cards.has(card));
              if (deducedCard && !agentKnownCards.has(deducedCard)) {
                  this.logger.debug(`Rule 2: ${agent.name} holds 2/3 cards and can deduce ${deducedCard}`);
                  newlyDeducedCards.add(deducedCard);
                  agentKnownCards.add(deducedCard);
              }
          }
        }
        
      // Rule 3: If this agent is the challenger and showed a card, they can deduce that card
      if (agent.name === challengeResult.challengingAgent && challengeResult.cardToShow) {
          const shownCard = challengeResult.cardToShow;
          if (!agentKnownCards.has(shownCard)) {
              this.logger.debug(`Rule 3: ${agent.name} showed ${shownCard} and can deduce it`);
              newlyDeducedCards.add(shownCard);
              agentKnownCards.add(shownCard);
          }
      }

      const finalDeductions = Array.from(newlyDeducedCards);
      this.logger.debug(`Final deductions for ${agent.name}: ${finalDeductions.join(', ') || 'None'}`);
      return finalDeductions;
  }

  async handleGameOver(reason, winningAgent) {
    if (this.isOver) return; // Prevent multiple calls
    this.isOver = true;
    this.winner = winningAgent; // Ensure winner is set here

    const reasonMessage = reason || 'Game Over';
    const winnerName = winningAgent ? `${winningAgent.name} (${winningAgent.model})` : 'No winner';
    this.logger.log('info', `--- GAME OVER --- Reason: ${reasonMessage}. Winner: ${winnerName}.`);

    // Emit event for UI or other listeners
    if (this.io) {
        this.logger.log('info', 'Emitting game-over event via socket.io');
        this.io.emit('game-over', {
            winner: this.winner ? {
                name: this.winner.name,
                model: this.winner.model
            } : null,
            solution: this.solution,
            reason: reasonMessage
        });
    }
    // Note: Saving results is handled *after* the runGameLoop finishes in server.js
  }
  
  /**
   * Formats turn events into a narrative string for LLM context or logging.
   * Adapts the narrative based on the recipient agent's perspective.
   * 
   * @param {Agent} suggestingAgent - The agent who made the suggestion.
   * @param {Object} suggestion - The suggestion {suspect, weapon, room}.
   * @param {Object} challengeResult - {challenger, cardShown, noChallenge}.
   * @param {Agent | null} recipientAgent - The agent whose perspective we're taking (null for general context).
   * @returns {Array<string>} An array of strings describing the turn events.
   */
  formatTurnEvents(suggestingAgent, suggestion, challengeResult, recipientAgent) {
    const events = [];
    const suggesterName = suggestingAgent.name;

    // Event 1: Suggestion
    events.push(`${suggesterName} suggested: ${suggestion.suspect}, ${suggestion.weapon}, ${suggestion.room}.`);
    
    // Event 2: Challenge Result
    const cardActuallyShown = challengeResult.cardToShow; // Capture the value safely
    const challengerName = challengeResult.challengingAgent;
    const challengerDisplayName = challengerName || 'An unknown agent'; // Handle potential null challenger if logic changes

    if (!challengeResult.canChallenge || !challengerName || !cardActuallyShown) {
        // Use canChallenge for clarity, also check if challenger/card are validly present
        events.push(`No one could disprove ${suggesterName}'s suggestion.`);
        } else {
        // Determine the narrative based on the recipient agent's perspective
        if (!recipientAgent) {
             // General context (no specific recipient)
             events.push(`${challengerDisplayName} challenged and showed a card to ${suggesterName}.`);
        } else if (recipientAgent.name === suggesterName) {
            // Perspective of the suggesting agent - USE cardActuallyShown
            events.push(`${challengerDisplayName} showed you the card: ${cardActuallyShown}.`);
        } else if (recipientAgent.name === challengerName) {
            // Perspective of the challenging agent - USE cardActuallyShown
            events.push(`You showed ${suggesterName} the card: ${cardActuallyShown}.`);
        } else {
            // Perspective of an observer
            events.push(`${challengerDisplayName} showed a card to ${suggesterName} (you did not see the card).`);
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