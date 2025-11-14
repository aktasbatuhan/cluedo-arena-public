import { SUSPECTS, WEAPONS, ROOMS } from '../../../config/gameConstants.js';

/**
 * Builds prompts for different LLM tasks in the Cluedo game.
 * Centralizes all prompt engineering logic.
 */
export class PromptBuilder {
  /**
   * Builds a prompt for making a suggestion.
   *
   * @param {Object} agent - The agent making the suggestion
   * @param {Object} gameState - Current game state
   * @param {Object} memoryState - Formatted memory state
   * @returns {string} Prompt text
   */
  static buildSuggestionPrompt(agent, gameState, memoryState) {
    return `You are the Cluedo agent ${agent.name}. Your turn ${gameState.currentTurn}.
Your hand: ${Array.from(agent.cards).join(', ')}.
Your current location: ${agent.location || 'Unknown (must be in a room to suggest)'}.
Available Rooms: ${gameState.availableRooms.join(', ')}

Your knowledge:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
Eliminated Cards (Not in Solution): ${Array.from(agent.memory.eliminatedCards).join(', ') || 'None'}
Suspected Cards: ${JSON.stringify(Object.fromEntries(agent.memory.suspectedCards))}
Current Deductions Summary: ${memoryState.currentDeductions}
Turn History Highlights:
${memoryState.turnHistory.join('\n')}

Based on your knowledge and location (${agent.location}), make a strategic suggestion (suspect, weapon, room).
The suggested room MUST be your current location: ${agent.location}.
Your goal is to gain new information by forcing others to reveal cards. Choose a suggestion that includes cards you suspect might be the solution OR cards held by others. Avoid suggesting only cards you know are eliminated unless tactically necessary.

Respond ONLY with a YAML object in the following format. Provide concise reasoning.

suspect: <string, one of ${gameState.availableSuspects.join(' | ')}>
weapon: <string, one of ${gameState.availableWeapons.join(' | ')}>
room: <string, MUST be ${agent.location}>
reasoning: <string, your detailed thought process for this suggestion>`;
  }

  /**
   * Builds a prompt for updating memory based on turn events.
   *
   * @param {Object} agent - The agent whose memory is being updated
   * @param {Object} memory - Agent's memory object
   * @param {Array<string>} turnEvents - Events from the current turn
   * @param {number} currentTurn - Current turn number
   * @returns {string} Prompt text
   */
  static buildMemoryUpdatePrompt(agent, memory, turnEvents, currentTurn) {
    return `You are ${agent.name}. Analyze the events from your last turn (Turn ${currentTurn}) and update your memory and deductions.

WHAT IS A DEDUCTION:
A deduction is a card that you can definitively conclude is NOT part of the murder solution. Deduce cards when:
1. It's in your hand.
2. Another player shows it to you.
3. You can logically prove it must be held by a specific player or eliminated.

Your current knowledge:
Cards in my hand: ${Array.from(agent.cards).join(', ')}
Known Eliminated Cards: ${Array.from(memory.eliminatedCards).join(', ') || 'None'}
Your most recent memory note:
${memory.currentMemory || '(No previous memory)'}

Events from THIS turn:
${turnEvents.map(event => event.replace(agent.name, 'I').replace(/^I showed/, 'I showed').replace(/showed you/, 'showed me')).join('\n')}

Based ONLY on the information above, what new cards can you definitively deduce are NOT part of the solution?
Remember: A deduction must be 100% certain.

Respond ONLY with a YAML object in the following format. Provide a DETAILED summary.

newlyDeducedCards:
  - <string> # Card name, or empty list if none
reasoning: <string> # Explain exactly how you deduced each new card
memorySummary: <string> # DETAILED summary of your CURRENT understanding. Include ALL known eliminated cards (hand + deduced), suspicions, and key insights from the game history.`;
  }

  /**
   * Builds a prompt for evaluating a challenge.
   *
   * @param {Object} agent - The agent responding to the challenge
   * @param {Object} suggestion - The suggestion being challenged
   * @param {Array<string>} matchingCards - Cards the agent has that match the suggestion
   * @param {Object} memoryState - Formatted memory state
   * @returns {string} Prompt text
   */
  static buildChallengePrompt(agent, suggestion, matchingCards, memoryState) {
    return `You are ${agent.name}. You received a suggestion: ${suggestion.suspect}, ${suggestion.weapon}, ${suggestion.room}.
You hold the following matching card(s): ${matchingCards.join(', ')}.

Your current knowledge:
Known cards held: ${Array.from(agent.cards).join(', ') || 'None'}
Eliminated Cards (Not in Solution): ${Array.from(agent.memory.eliminatedCards).join(', ') || 'None'}
Current Deductions Summary: ${memoryState.currentDeductions}

Choose ONE card from your matching cards (${matchingCards.join(', ')}) to show to the suggester.
Consider which card reveals the least about your overall hand and deductions.

Respond ONLY with a YAML object in the following format.

cardToShow: <string, must be one of: ${matchingCards.join(' | ')}>
reasoning: <string, briefly explain your choice>`;
  }

  /**
   * Builds a prompt for considering an accusation.
   *
   * @param {Object} agent - The agent considering the accusation
   * @param {Object} gameState - Current game state
   * @param {Object} memoryState - Formatted memory state
   * @returns {string} Prompt text
   */
  static buildAccusationPrompt(agent, gameState, memoryState) {
    const currentSuggestion = gameState.currentSuggestion;
    const currentChallengeResult = gameState.currentChallengeResult;

    let currentTurnEventsString = 'No suggestion made this turn.';
    if (currentSuggestion) {
      currentTurnEventsString = `This turn, you suggested: ${currentSuggestion.suspect}, ${currentSuggestion.weapon}, ${currentSuggestion.room}.\n`;
      if (currentChallengeResult && currentChallengeResult.cardToShow) {
        currentTurnEventsString += `Result: ${currentChallengeResult.challengingAgent} showed you the card: ${currentChallengeResult.cardToShow}.`;
      } else {
        currentTurnEventsString += `Result: NO ONE could challenge your suggestion.`; // Highlight this crucial outcome
      }
    }

    return `You are ${agent.name}. Turn ${gameState.currentTurn}. Decide if you should make a final accusation to win.

Your knowledge:
- Your Hand: ${Array.from(agent.cards).join(', ') || 'None'}
- Eliminated Cards (Not in Solution): ${Array.from(agent.memory.eliminatedCards).join(', ') || 'None'}
- Suspected Cards: ${JSON.stringify(Object.fromEntries(agent.memory.suspectedCards))}
- Current Deductions Summary: ${memoryState.currentDeductions}
- Turn History Highlights:
${memoryState.turnHistory.join('\n')}

Current Turn Events (Turn ${gameState.currentTurn}):
${currentTurnEventsString}

CLUEDO LOGIC REMINDERS:
1. No challenge to a suggestion implies suggested cards MIGHT be the solution.
2. A challenge proves AT LEAST ONE suggested card is NOT the solution.
3. Elimination: If 5/6 suspects are known, the last one IS the solution suspect.
4. You can risk an accusation without 100% certainty based on strong evidence.

Consider accusing if:
- You have strong evidence/elimination for all 3 solution components.
- A key suggestion was unchallenged.

Respond ONLY with a YAML object in the following format.
If shouldAccuse is true, provide your deduced solution.
If shouldAccuse is false, provide null for accusation components.

shouldAccuse: <boolean>
accusation:
  suspect: <string | null, one of ${SUSPECTS.join(' | ')} or null>
  weapon: <string | null, one of ${WEAPONS.join(' | ')} or null>
  room: <string | null, one of ${ROOMS.join(' | ')} or null>
reasoning: <string, explain your decision and confidence level>`;
  }
}
