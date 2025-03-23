import { LLMService } from '../services/llm.js';
import { Agent } from './Agent.js';
import { HumanAgent } from './HumanAgent.js';
import { GameResult } from './GameResult.js';
import { MODEL_LIST } from '../services/llm.js';
import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
let ioInstance = null;

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

  getRemainingCards() {
    const allCards = [...this.SUSPECTS, ...this.WEAPONS, ...this.ROOMS];
    return allCards.filter(card => 
      card !== this.solution.suspect && 
      card !== this.solution.weapon && 
      card !== this.solution.room
    );
  }

  distributeCards(cards) {
    const shuffled = [...cards].sort(() => Math.random() - 0.5);
    const dealtCards = [];
    const cardsPerAgent = Math.floor(shuffled.length / 6);
    
    for (let i = 0; i < 6; i++) {
      dealtCards.push(
        shuffled.slice(i * cardsPerAgent, (i + 1) * cardsPerAgent)
      );
    }
    
    return dealtCards;
  }

  getRandomElement(array) {
    return array[Math.floor(Math.random() * array.length)];
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
    const agent = this.agents[this.activeAgentIndex];
    const turnResult = {
      agent: agent.name,
      suggestion: null,
      accusationDecision: null,
      theory: null
    };

    if (agent.hasLost) {
      this.nextTurn();
      return turnResult;
    }

    console.log(`\n=== TURN ${this.currentTurn} === [Agent: ${this.activeAgentIndex}]`);
    console.log(`Processing ${agent.name}'s turn...`);
    
    // Start timing the turn
    console.time(`[Turn] ${agent.name}'s full turn`);

    try {
      // 1. Make suggestion first
      console.log(`${agent.name} is making a suggestion...`);
      const suggestion = await agent.makeSuggestion(this.getGameState());
      turnResult.suggestion = suggestion;
      
      if (suggestion) {
        console.log(`Suggestion from ${agent.name}:`, suggestion);
        this.emit('suggestion', {
          agent: agent.name,
          suggestion: suggestion,
          reasoning: suggestion.reasoning || 'No reasoning provided',
          timestamp: new Date()
        });
      }

      // 2. Process challenges
      console.log('Processing challenges...');
      const challengeResult = await this.processChallenges(agent, suggestion);
      
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
      console.log(`${agent.name} is considering an accusation...`);
      const accusationDecision = await agent.considerAccusation();
      turnResult.accusationDecision = accusationDecision;
      console.log('Accusation decision:', accusationDecision);
      
      // 4. If agent decides to make accusation, process it
      if (accusationDecision.shouldAccuse) {
        console.log(`${agent.name} is making an accusation!`);
        this.emit('accusation', {
          agent: agent.name,
          accusation: accusationDecision.accusation,
          reasoning: accusationDecision.reasoning,
          confidence: accusationDecision.confidence,
          timestamp: new Date()
        });

        const isCorrect = this.checkAccusation(
          agent,
          accusationDecision.accusation.suspect,
          accusationDecision.accusation.weapon,
          accusationDecision.accusation.room
        );

        if (!isCorrect) {
          agent.setLost();
          this.activePlayers--;
          console.log(`${agent.name} made an incorrect accusation and was eliminated`);
        } else {
          this.winner = agent;
          console.log(`${agent.name} won with a correct accusation!`);
          return turnResult;
        }
      }
      

      // 6. Update all agents' memories with turn events
      await this.updateAgentMemories({
        turnNumber: this.currentTurn,
        activeAgent: agent.name,
        suggestion,
        challengeResult,
        accusationResult: accusationDecision.shouldAccuse ? {
          wasCorrect: this.winner === agent,
          accusation: accusationDecision.accusation
        } : null
      });

      // End timing the turn
      console.timeEnd(`[Turn] ${agent.name}'s full turn`);

      // Emit updated game state
      if (this.io) {
        this.io.emit('game-state', this.getGameSummary());
      }

      this.nextTurn();
      
      // Add delay between turns
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      return turnResult;

    } catch (error) {
      console.error('Error processing turn:', error);
      // End timing even if there's an error
      console.timeEnd(`[Turn] ${agent.name}'s full turn`);
      this.nextTurn();
      return turnResult;
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

  async processChallenges(agent, suggestion) {
    try {
      const challengers = this.agents
        .filter(a => 
          !a.hasLost && 
          a !== agent
        )
        .sort((a, b) => 
          (a.position === suggestion.room ? -1 : 0) - 
          (b.position === suggestion.room ? -1 : 0)
        );

      for (const challenger of challengers) {
        const challengeResult = await challenger.evaluateChallenge(suggestion);
        
        if (challengeResult.canChallenge) {
          // Validate the challenger actually has the card
          if (!challenger.cards.has(challengeResult.cardToShow)) {
            console.error(`Invalid challenge from ${challenger.name}: Does not possess ${challengeResult.cardToShow}`);
            continue;
          }

          return {
            canChallenge: true,
            challengingAgent: challenger.name,
            cardToShow: challengeResult.cardToShow
          };
        }
      }

      return { canChallenge: false };
    } catch (error) {
      console.error('Challenge processing failed:', error);
      return { canChallenge: false };
    }
  }

  formatNarrativeSuggestion(agentName, suggestion) {
    return `${agentName} suggests that the crime was committed by ${suggestion.suspect} in the ${suggestion.room} with the ${suggestion.weapon}.`;
  }

  formatNarrativeChallenge(agentName, card) {
    return `${agentName} shows a card to disprove the suggestion.`;
  }

  async updateAgentMemories(turnEvents) {
    try {
      // Update each agent's memory with complete turn data
      for (const agent of this.agents) {
        if (!agent.hasLost) {
          try {
            await agent.updateMemory(turnEvents);
          } catch (error) {
            console.error(`Failed to update ${agent.name}'s memory:`, error);
          }
        }
      }
    } catch (error) {
      console.error('Failed to update agent memories:', error);
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

      // Create and set solution
      this.solution = this.createSolution();
      
      // Log solution
      console.log(`Solution: ${this.solution.suspect} in the ${this.solution.room} with the ${this.solution.weapon}`);
      
      // Emit solution to clients
      if (this.io) {
        this.io.emit('game-solution', this.solution);
      }

      // Get remaining cards
      const remainingCards = this.getRemainingCards();
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
  checkAccusation(agent, suspectGuess, weaponGuess, roomGuess) {
    // Check if accusation matches solution first
    const isCorrect = 
      this.solution.suspect === suspectGuess &&
      this.solution.weapon === weaponGuess &&
      this.solution.room === roomGuess;

    // Set winner if correct before any logging
    if (isCorrect) {
      this.winner = agent;
    } else {
      agent.setLost();
      this.activePlayers--;
    }

    // Track the accusation
    this.accusations.push({
      agent: agent.name,
      accusation: {
        suspect: suspectGuess,
        weapon: weaponGuess,
        room: roomGuess
      },
      isCorrect,
      turnNumber: this.currentTurn
    });

    // Now emit events in the correct order with proper messages
    if (isCorrect) {
      this.logEvent({
        type: 'GAME_OVER',
        agent: agent.name,
        accusation: {
          suspect: suspectGuess,
          weapon: weaponGuess,
          room: roomGuess
        },
        message: `${agent.name} won with a correct accusation!`,
        timestamp: new Date()
      });
    } else {
      this.logEvent({
        type: 'ACCUSATION_FAILED',
        agent: agent.name,
        accusation: {
          suspect: suspectGuess,
          weapon: weaponGuess,
          room: roomGuess
        },
        message: `${agent.name} made an incorrect accusation and was eliminated.`,
        timestamp: new Date()
      });
    }

    return isCorrect;
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
}