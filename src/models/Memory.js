export class Memory {
  constructor(agentId) {
    this.agentId = agentId;
    
    // Card state tracking
    this.knownCards = new Set();           // Cards in hand
    this.suspectedCards = new Map();       // Cards with confidence levels
    this.eliminatedCards = new Set();      // Cards proven not in solution
    
    // Player-specific negative constraints
    this.playerNegativeConstraints = new Map(); // playerName -> Set<cardName>
    
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

  /**
   * Checks if memory indicates a specific player does NOT have a specific card.
   * @param {string} playerName - The name of the player.
   * @param {string} cardName - The name of the card.
   * @returns {boolean} - True if the player is known not to have the card, false otherwise.
   */
  doesPlayerNotHaveCard(playerName, cardName) {
    const constraints = this.playerNegativeConstraints.get(playerName);
    return constraints ? constraints.has(cardName) : false;
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
    
    // --- REVERTED --- Combine known cards (agent's hand) and explicitly eliminated cards for the LLM
    // const allEliminated = new Set([...this.knownCards, ...this.eliminatedCards]);
    
    return {
      knownInformation: {
        myCards: Array.from(this.knownCards), // Still useful to show the agent's actual hand separately
        eliminatedCards: Array.from(this.eliminatedCards), // Provide ONLY deduced/shown cards, not own hand
        suspectedCards: Object.fromEntries(this.suspectedCards),
        deducedCards: { // Keep for now, might simplify later
          suspects: deducedSuspects,
          weapons: deducedWeapons,
          rooms: deducedRooms
        }
      },
      currentDeductions: (() => {
        if (!this.currentMemory || this.currentMemory.trim() === "") {
            return "(No previous memory summary)";
        }
        // Find the last occurrence of the timestamp pattern
        const lastUpdateIndex = this.currentMemory.lastIndexOf('\n\n[');
        if (lastUpdateIndex === -1) {
            // If the pattern isn't found (e.g., only initial state), return the whole string
            return this.currentMemory;
        }
        // Extract the part from the last timestamp onwards
        // We add 2 to skip the initial '\n\n'
        const latestUpdate = this.currentMemory.substring(lastUpdateIndex + 2);
        // Find the end of the timestamp/header line
        const headerEndIndex = latestUpdate.indexOf('\n'); 
        if (headerEndIndex !== -1) {
            // Return just the summary content after the header line
            return latestUpdate.substring(headerEndIndex + 1).trim(); 
        } else {
            // If somehow only the header exists, return it (unlikely)
            return latestUpdate.trim();
        }
      })(),

      // Simplify turn history formatting for LLM - Use raw events
      turnHistory: this.memoryHistory.map(entry => {
          let eventString = `Turn ${entry.turnNumber || '?'}:`;
          if (entry.suggestion) {
              eventString += ` ${entry.activeAgent || 'Agent'} suggested ${entry.suggestion.suspect}, ${entry.suggestion.weapon}, ${entry.suggestion.room}.`;
          }
          if (entry.challengeResult?.canChallenge) {
              eventString += ` ${entry.challengeResult.challengingAgent} showed a card.`; // Simplified
          } else if (entry.suggestion) {
              eventString += ` No challenge.`;
          }
          // We can add accusation info here later if needed
          return eventString;
      })
      // OLD formatting with LLM summaries:
      // turnHistory: this.memoryHistory.map(entry => {
      //   // Handle both old format turnHistory entries and newer ones with summary
      //   if (entry.summary) {
      //     // Add safety check for summary type
      //     const summary = typeof entry.summary === 'string' ? entry.summary : String(entry.summary || '');
      //     // Limit summary length for display, but keep full history accessible
      //     return `Turn ${entry.turnNumber || '?'}: ${summary.substring(0, 150)}${summary.length > 150 ? '...' : ''}`;
      //   } else if (entry.suggestion) {
      //     return `Turn ${entry.turnNumber}: ${entry.activeAgent} suggested ${entry.suggestion?.suspect}, ${entry.suggestion?.weapon}, ${entry.suggestion?.room}` +
      //     (entry.challengeResult?.canChallenge ? 
      //       ` → ${entry.challengeResult.challengingAgent} showed ${entry.challengeResult.cardToShow}` : 
      //       ' → No challenge');
      //   } else {
      //     return `Turn ${entry.turnNumber || '?'}: Event recorded`;
      //   }
      // })
    };
  }

  /**
   * Records the key structured events of a turn into the memory history.
   * This history is used by formatMemoryForLLM to generate the simplified turn history string.
   *
   * @param {number} turnNumber - The game turn number.
   * @param {string} activeAgent - The name of the agent who made the suggestion (or null if none).
   * @param {object} suggestion - The suggestion object {suspect, weapon, room} (or null if none).
   * @param {object} challengeResult - The challenge result object {canChallenge, challengingAgent, cardToShow} (or null if none).
   */
  recordTurnEvents(turnNumber, activeAgent, suggestion, challengeResult) {
    this.memoryHistory.push({
      turnNumber: turnNumber || this.memoryHistory.length + 1, // Use provided turn number or estimate
      timestamp: Date.now(),
      activeAgent: activeAgent, // Agent who made the suggestion
      suggestion: suggestion ? { // Store the core suggestion details
        suspect: suggestion.suspect,
        weapon: suggestion.weapon,
        room: suggestion.room
      } : null,
      challengeResult: challengeResult ? { // Store the core challenge outcome
        canChallenge: challengeResult.canChallenge,
        challengingAgent: challengeResult.challengingAgent
        // Do NOT store cardToShow here - it's secret to most!
      } : null
    });
    this.lastUpdated = Date.now(); // Update timestamp when history is added
  }

  // Update memory with new turn information
  async updateMemory(turnEvents) {
    // Create and store memory entry
    // TODO: Consider if this method should primarily update structured state
    // and leave the LLM call/summary append to a separate flow.
    // For now, it updates some structured state based on direct reveals.

    // --- Update Structured State based on Events ---
    let suggestionCards = [];

    for (const event of turnEvents) {
      if (event.type === 'suggestion') {
         suggestionCards = [event.suspect, event.weapon, event.room];
      }
      // Direct card shown TO this agent (or by this agent)
      else if (event.type === 'cardShown' && event.card) {
         if (event.from === this.agentId) {
             // We showed the card
             this.addKnownCard(event.card);
         } else if (event.to === this.agentId) {
             // Card shown TO us
             this.eliminateCard(event.card); // Add to general eliminated set
             // We also know the specific player who showed it
             // Future enhancement: Store {card: player} mapping?
         }
      }
      // Challenge outcome (might be hidden)
      else if (event.type === 'challenge') {
          const challengerName = event.challengingAgent;
          const couldChallenge = event.canChallenge;

          if (challengerName && !couldChallenge && suggestionCards.length === 3) {
              // Player could NOT challenge the suggestion
              // Add negative constraints: challenger does not have these suggested cards
              if (!this.playerNegativeConstraints.has(challengerName)) {
                  this.playerNegativeConstraints.set(challengerName, new Set());
              }
              const constraints = this.playerNegativeConstraints.get(challengerName);
              suggestionCards.forEach(card => constraints.add(card));
              // logger.debug(`Added negative constraints for ${challengerName}: does not have ${suggestionCards.join(', ')}`);
          }
      }
    }
    // -----------------------------------------------

    // Note: The memory history and lastUpdated are now handled
    // primarily by the separate .update() method called from LLMService
    // when the LLM provides a summary.
    // Keep this basic update for direct reveals, but the main history
    // and memory string update happens via .update().

    // // Old logic - Replaced by Memory.update() called from LLMService
    // const memoryEntry = this.createMemoryEntry(turnEvents);
    // this.memoryHistory.push(memoryEntry);
    // this.lastUpdated = Date.now();
  }

  reset() {
    this.currentMemory = "";
    this.memoryHistory = [];
    this.suspectedCards.clear();
    this.eliminatedCards.clear();
    this.playerNegativeConstraints.clear(); // Clear negative constraints
    this.confidence.suspects.clear();
    this.confidence.weapons.clear();
    this.confidence.rooms.clear();
  }
  
  // Memory maintenance method to handle cleanup and optimization
  maintain() {
    // Limit memory history size to prevent excessive growth
    if (this.memoryHistory.length > 100) {
      this.memoryHistory = this.memoryHistory.slice(-100);
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
   * the LLM's reasoning and deductions into the agent's cumulative memory
   * 
   * @param {string} summary - The summary generated by the LLM
   * @param {string[]} newlyDeducedCards - Array of card names deduced by the LLM this turn
   * @param {string} reasoning - The reasoning provided by the LLM for the deductions
   */
  async update(summary, newlyDeducedCards = [], reasoning = '') {
    // Don't update with empty summary, but still process deductions
    if (summary && summary !== "") {
      // Add timestamp to the summary
      const timestamp = new Date().toISOString();
      const formattedSummary = `\n\n[${timestamp}] MEMORY UPDATE:\n${summary}`;
      
      // Append new summary to existing memory, ensuring cumulative buildup
      if (!this.currentMemory) {
        this.currentMemory = formattedSummary;
      } else {
        this.currentMemory += formattedSummary;
      }
    }

    // --- CRITICAL FIX: Update the eliminatedCards Set --- 
    if (Array.isArray(newlyDeducedCards)) {
      for (const card of newlyDeducedCards) {
        if (card && typeof card === 'string') { // Basic validation
          this.eliminateCard(card); // Use existing method to update the Set
          // console.log(`${this.agentId} eliminated card via LLM deduction: ${card}`); // Optional debug log
        }
      }
    }
    // --------------------------------------------------
    
    // REMOVE history push from here - it's handled by recordTurnEvents now
    /*
    this.memoryHistory.push({
      turnNumber: this.memoryHistory.length + 1, // Use history length for turn number
      timestamp: Date.now(),
      summary: summary || '(No summary provided)', // Use summary if available
      deduced: newlyDeducedCards, // Store deductions in history
      reasoning: reasoning // Store reasoning
    });
    */

    // Update timestamp
    this.lastUpdated = Date.now();
  }
} 