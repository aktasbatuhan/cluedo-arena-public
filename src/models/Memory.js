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
    // Get lists of deduced cards by category
    const deducedSuspects = [];
    const deducedWeapons = [];
    const deducedRooms = [];
    
    // Add cards that have been explicitly deduced from eliminated cards and suspicions
    for (const [card, confidence] of this.suspectedCards.entries()) {
      if (confidence > 0.8) { // High confidence threshold
        if (this.game?.SUSPECTS?.includes(card)) deducedSuspects.push(card);
        else if (this.game?.WEAPONS?.includes(card)) deducedWeapons.push(card);
        else if (this.game?.ROOMS?.includes(card)) deducedRooms.push(card);
      }
    }
    
    return {
      knownInformation: {
        myCards: Array.from(this.knownCards),
        eliminatedCards: Array.from(this.eliminatedCards),
        suspectedCards: Object.fromEntries(this.suspectedCards),
        deducedCards: {
          suspects: deducedSuspects,
          weapons: deducedWeapons,
          rooms: deducedRooms
        },
        confidence: {
          suspects: Object.fromEntries(this.confidence.suspects),
          weapons: Object.fromEntries(this.confidence.weapons),
          rooms: Object.fromEntries(this.confidence.rooms)
        }
      },
      currentDeductions: this.currentMemory,
      turnHistory: this.memoryHistory.slice(-5).map(entry => {
        // Handle both old format turnHistory entries and newer ones with summary
        if (entry.summary) {
          // Add safety check for summary type
          const summary = typeof entry.summary === 'string' ? entry.summary : String(entry.summary || '');
          return `Turn ${entry.turnNumber || '?'}: ${summary.substring(0, 100)}${summary.length > 100 ? '...' : ''}`;
        } else if (entry.suggestion) {
          return `Turn ${entry.turnNumber}: ${entry.activeAgent} suggested ${entry.suggestion?.suspect}, ${entry.suggestion?.weapon}, ${entry.suggestion?.room}` +
          (entry.challengeResult?.canChallenge ? 
            ` → ${entry.challengeResult.challengingAgent} showed ${entry.challengeResult.cardToShow}` : 
            ' → No challenge');
        } else {
          return `Turn ${entry.turnNumber || '?'}: Event recorded`;
        }
      })
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
    
    // Limit the size of the currentMemory to prevent excessive growth
    // This is a simple approach - a more sophisticated approach would involve
    // summarizing the older memories or keeping only the most important deductions
    const MAX_MEMORY_LENGTH = 10000; // About 10KB of text, adjust as needed
    
    if (this.currentMemory && this.currentMemory.length > MAX_MEMORY_LENGTH) {
      // Keep most recent memories - the first section might contain initial setup
      // and the last section contains recent updates
      const initialSection = this.currentMemory.substring(0, 1000); // Keep first 1000 chars
      const recentSection = this.currentMemory.substring(this.currentMemory.length - MAX_MEMORY_LENGTH + 1000);
      
      this.currentMemory = initialSection + 
        "\n\n[MEMORY TRUNCATED TO SAVE SPACE]\n\n" + 
        recentSection;
    }
    
    // Ensure memory state is consistent
    if (!this.currentMemory) {
      this.currentMemory = "";
    }
    
    // Update timestamp
    this.lastUpdated = Date.now();
  }

  /**
   * Updates memory with the LLM-generated summary
   * This is called from LLMService.updateMemory to integrate
   * the LLM's reasoning into the agent's cumulative memory
   * 
   * @param {string} turnEvents - The formatted turn events
   * @param {string} summary - The summary generated by the LLM
   */
  async update(turnEvents, summary) {
    // Don't update with empty summary
    if (!summary || summary === "") return;

    // Add timestamp to the summary
    const timestamp = new Date().toISOString();
    const formattedSummary = `\n\n[${timestamp}] MEMORY UPDATE:\n${summary}`;
    
    // Append new summary to existing memory, ensuring cumulative buildup
    if (!this.currentMemory) {
      this.currentMemory = formattedSummary;
    } else {
      this.currentMemory += formattedSummary;
    }
    
    // Also add a simplified entry to memory history for long-term tracking
    this.memoryHistory.push({
      turnNumber: this.memoryHistory.length + 1,
      timestamp: Date.now(),
      summary: summary,
      turnEvents: turnEvents
    });

    // Update timestamp
    this.lastUpdated = Date.now();
  }
} 