import { Agent } from './Agent.js';
import { Memory } from './Memory.js';

export class HumanAgent extends Agent {
  constructor(name, cards) {
    super(name, cards, 'human');
    this.memory = new Memory(name);
    this.memory.addKnownCards(cards);
    this.pendingActions = [];
  }

  async updateMemory(turnData) {
    // Handle memory updates without using LLM
    if (turnData.type === 'SHOWN_CARD') {
      this.memory.shownCards.push({
        card: turnData.card,
        agent: turnData.agent,
        timestamp: turnData.timestamp
      });
      this.memory.seenCards.add(turnData.card);
    }

    if (turnData.type === 'CHALLENGE_OCCURRED') {
      this.memory.challenges.push({
        suggestingAgent: turnData.suggestingAgent,
        challengingAgent: turnData.challengingAgent,
        timestamp: turnData.timestamp
      });
    }

    // Return a simplified memory update response
    return {
      deductions: [],
      memory: this.memory
    };
  }

  async makeSuggestion() {
    return new Promise((resolve) => {
      this.pendingActions.push({ type: 'suggestion', resolve });
    });
  }

  async considerAccusation() {
    return new Promise((resolve) => {
      this.pendingActions.push({ type: 'accusation', resolve });
    });
  }

  async evaluateChallenge(suggestion) {
    // Check if we can disprove the suggestion with our cards
    const matchingCard = this.cards.find(card => 
      card === suggestion.suspect ||
      card === suggestion.weapon ||
      card === suggestion.room
    );

    return {
      canChallenge: !!matchingCard,
      cardToShow: matchingCard
    };
  }
} 