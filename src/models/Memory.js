export class Memory {
  constructor(agentId) {
    this.agentId = agentId;
    
    // Card state tracking
    this.knownCards = new Set();           // Cards in hand
    this.suspectedCards = new Map();       // Cards with confidence levels
    this.eliminatedCards = new Set();      // Cards proven not in solution
    
    // Memory state
    this.currentMemory = "";               // Current deductions
    this.memoryHistory = [];               // History of turn events
    this.lastUpdated = Date.now();

    // Confidence tracking per category
    this.confidence = {
      suspects: new Map(),
      weapons: new Map(),
      rooms: new Map()
    };
  }

  // Card state management methods
  addKnownCard(card) {
    this.knownCards.add(card);
    this.suspectedCards.delete(card);
    this.eliminatedCards.delete(card);
  }

  updateSuspicion(card, confidence) {
    if (!this.knownCards.has(card) && !this.eliminatedCards.has(card)) {
      this.suspectedCards.set(card, confidence);
    }
  }

  eliminateCard(card) {
    this.eliminatedCards.add(card);
    this.suspectedCards.delete(card);
  }

  // Create structured memory entry for a turn
  createMemoryEntry(turnData) {
    return {
      turnNumber: turnData.turnNumber,
      timestamp: Date.now(),
      activeAgent: turnData.activeAgent,
      suggestion: turnData.suggestion,
      challengeResult: turnData.challengeResult,
      deductions: turnData.deductions || [],
      memoryState: this.currentMemory
    };
  }

  // Format memory for LLM consumption
  async formatMemoryForLLM() {
    return {
      knownInformation: {
        myCards: Array.from(this.knownCards),
        eliminatedCards: Array.from(this.eliminatedCards),
        suspectedCards: Object.fromEntries(this.suspectedCards),
        confidence: {
          suspects: Object.fromEntries(this.confidence.suspects),
          weapons: Object.fromEntries(this.confidence.weapons),
          rooms: Object.fromEntries(this.confidence.rooms)
        }
      },
      currentDeductions: this.currentMemory,
      turnHistory: this.memoryHistory.slice(-5).map(entry => 
        `Turn ${entry.turnNumber}: ${entry.activeAgent} suggested ${entry.suggestion?.suspect}, ${entry.suggestion?.weapon}, ${entry.suggestion?.room}` +
        (entry.challengeResult?.canChallenge ? 
          ` → ${entry.challengeResult.challengingAgent} showed ${entry.challengeResult.cardToShow}` : 
          ' → No challenge')
      )
    };
  }

  // Update memory with new turn information
  async updateMemory(turnEvents) {
    // Create and store memory entry
    const memoryEntry = this.createMemoryEntry(turnEvents);
    this.memoryHistory.push(memoryEntry);

    // Update card states based on turn events
    if (turnEvents.challengeResult?.canChallenge) {
      const shownCard = turnEvents.challengeResult.cardToShow;
      if (turnEvents.challengeResult.challengingAgent === this.agentId) {
        // If we showed the card, it's known
        this.addKnownCard(shownCard);
      } else {
        // If someone else showed a card, it's eliminated from solution
        this.eliminateCard(shownCard);
      }
    }

    // Update last modified timestamp
    this.lastUpdated = Date.now();
  }

  reset() {
    this.currentMemory = "";
    this.memoryHistory = [];
    this.suspectedCards.clear();
    this.eliminatedCards.clear();
    this.confidence.suspects.clear();
    this.confidence.weapons.clear();
    this.confidence.rooms.clear();
  }
  
  // Memory maintenance method to handle cleanup and optimization
  maintain() {
    // Limit memory history size to prevent excessive growth
    if (this.memoryHistory.length > 30) {
      this.memoryHistory = this.memoryHistory.slice(-30);
    }
    
    // Ensure memory state is consistent
    if (!this.currentMemory) {
      this.currentMemory = "";
    }
    
    // Update timestamp
    this.lastUpdated = Date.now();
  }
} 