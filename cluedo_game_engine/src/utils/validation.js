/**
 * Input validation utilities for game logic and LLM responses.
 * @module utils/validation
 */

import { SUSPECTS, WEAPONS, ROOMS } from '../config/gameConstants.js';
import { GameValidationError, LLMValidationError } from './errors.js';

/**
 * Validates a card name against the game's valid cards.
 *
 * @param {string} card - Card name to validate
 * @param {Object} options - Validation options
 * @param {boolean} [options.allowNull=false] - Whether null is acceptable
 * @param {string} [options.context=''] - Context for error messages
 * @returns {boolean} True if valid
 * @throws {GameValidationError} If card is invalid
 */
export function validateCard(card, { allowNull = false, context = '' } = {}) {
  if (card === null || card === undefined) {
    if (allowNull) return true;
    throw new GameValidationError('Card cannot be null or undefined', { context });
  }

  const allCards = [...SUSPECTS, ...WEAPONS, ...ROOMS];

  if (!allCards.includes(card)) {
    throw new GameValidationError(`Invalid card: ${card}`, {
      context: {
        card,
        validCards: allCards,
        userContext: context,
      },
    });
  }

  return true;
}

/**
 * Validates a suggestion object.
 *
 * @param {Object} suggestion - Suggestion to validate
 * @param {string} suggestion.suspect - Suspect name
 * @param {string} suggestion.weapon - Weapon name
 * @param {string} suggestion.room - Room name
 * @param {string} [suggestion.reasoning] - Optional reasoning
 * @param {Object} options - Validation options
 * @param {string} [options.currentRoom] - Current room (suggestion room must match)
 * @param {string} [options.agentName] - Agent making suggestion
 * @returns {boolean} True if valid
 * @throws {GameValidationError} If suggestion is invalid
 */
export function validateSuggestion(suggestion, { currentRoom, agentName } = {}) {
  if (!suggestion || typeof suggestion !== 'object') {
    throw new GameValidationError('Suggestion must be an object', {
      agentName,
      context: { suggestion },
    });
  }

  const { suspect, weapon, room, reasoning } = suggestion;

  // Validate required fields exist
  if (!suspect || !weapon || !room) {
    throw new GameValidationError('Suggestion must include suspect, weapon, and room', {
      agentName,
      context: { suggestion },
    });
  }

  // Validate each card
  validateCard(suspect, { context: `suggestion.suspect for ${agentName}` });
  validateCard(weapon, { context: `suggestion.weapon for ${agentName}` });
  validateCard(room, { context: `suggestion.room for ${agentName}` });

  // Validate card types
  if (!SUSPECTS.includes(suspect)) {
    throw new GameValidationError(`Invalid suspect: ${suspect}`, {
      agentName,
      context: { suspect, validSuspects: SUSPECTS },
    });
  }

  if (!WEAPONS.includes(weapon)) {
    throw new GameValidationError(`Invalid weapon: ${weapon}`, {
      agentName,
      context: { weapon, validWeapons: WEAPONS },
    });
  }

  if (!ROOMS.includes(room)) {
    throw new GameValidationError(`Invalid room: ${room}`, {
      agentName,
      context: { room, validRooms: ROOMS },
    });
  }

  // Validate room matches current location if provided
  if (currentRoom && room !== currentRoom) {
    throw new GameValidationError(`Suggestion room (${room}) must match current room (${currentRoom})`, {
      agentName,
      context: { suggestedRoom: room, currentRoom },
    });
  }

  // Validate reasoning if provided
  if (reasoning !== undefined && typeof reasoning !== 'string') {
    throw new GameValidationError('Reasoning must be a string', {
      agentName,
      context: { reasoning },
    });
  }

  return true;
}

/**
 * Validates an accusation object.
 *
 * @param {Object} accusation - Accusation to validate
 * @param {boolean} accusation.shouldAccuse - Whether to make an accusation
 * @param {Object} accusation.accusation - Accusation details
 * @param {string|null} accusation.accusation.suspect - Suspect name or null
 * @param {string|null} accusation.accusation.weapon - Weapon name or null
 * @param {string|null} accusation.accusation.room - Room name or null
 * @param {string} [accusation.reasoning] - Optional reasoning
 * @param {Object} options - Validation options
 * @param {string} [options.agentName] - Agent making accusation
 * @returns {boolean} True if valid
 * @throws {GameValidationError} If accusation is invalid
 */
export function validateAccusation(accusation, { agentName } = {}) {
  if (!accusation || typeof accusation !== 'object') {
    throw new GameValidationError('Accusation must be an object', {
      agentName,
      context: { accusation },
    });
  }

  const { shouldAccuse, accusation: details, reasoning } = accusation;

  // Validate shouldAccuse is boolean
  if (typeof shouldAccuse !== 'boolean') {
    throw new GameValidationError('shouldAccuse must be a boolean', {
      agentName,
      context: { shouldAccuse },
    });
  }

  // Validate accusation details exist
  if (!details || typeof details !== 'object') {
    throw new GameValidationError('accusation details must be an object', {
      agentName,
      context: { details },
    });
  }

  const { suspect, weapon, room } = details;

  // If shouldAccuse is true, all fields must be valid cards
  if (shouldAccuse) {
    validateCard(suspect, { context: `accusation.suspect for ${agentName}` });
    validateCard(weapon, { context: `accusation.weapon for ${agentName}` });
    validateCard(room, { context: `accusation.room for ${agentName}` });

    if (!SUSPECTS.includes(suspect)) {
      throw new GameValidationError(`Invalid suspect in accusation: ${suspect}`, {
        agentName,
      });
    }

    if (!WEAPONS.includes(weapon)) {
      throw new GameValidationError(`Invalid weapon in accusation: ${weapon}`, {
        agentName,
      });
    }

    if (!ROOMS.includes(room)) {
      throw new GameValidationError(`Invalid room in accusation: ${room}`, {
        agentName,
      });
    }
  }

  // If shouldAccuse is false, fields should be null (but we're lenient here)
  // The LLM might still provide values even when not accusing

  return true;
}

/**
 * Validates a memory update response from LLM.
 *
 * @param {Object} memoryUpdate - Memory update to validate
 * @param {Array<string>} memoryUpdate.newlyDeducedCards - Cards deduced this turn
 * @param {string} memoryUpdate.reasoning - Reasoning for deductions
 * @param {string} memoryUpdate.memorySummary - Summary of current understanding
 * @param {Object} options - Validation options
 * @param {string} [options.agentName] - Agent name
 * @returns {boolean} True if valid
 * @throws {LLMValidationError} If memory update is invalid
 */
export function validateMemoryUpdate(memoryUpdate, { agentName } = {}) {
  if (!memoryUpdate || typeof memoryUpdate !== 'object') {
    throw new LLMValidationError('Memory update must be an object', {
      provider: 'unknown',
      agentName,
      taskType: 'memory_update',
      context: { memoryUpdate },
    });
  }

  const { newlyDeducedCards, reasoning, memorySummary } = memoryUpdate;

  // Validate newlyDeducedCards is an array
  if (!Array.isArray(newlyDeducedCards)) {
    throw new LLMValidationError('newlyDeducedCards must be an array', {
      provider: 'unknown',
      agentName,
      taskType: 'memory_update',
      context: { newlyDeducedCards },
    });
  }

  // Validate each deduced card
  const allCards = [...SUSPECTS, ...WEAPONS, ...ROOMS];
  for (const card of newlyDeducedCards) {
    if (typeof card !== 'string') {
      throw new LLMValidationError(`Deduced card must be a string: ${card}`, {
        provider: 'unknown',
        agentName,
        taskType: 'memory_update',
        context: { card, newlyDeducedCards },
      });
    }

    if (!allCards.includes(card)) {
      throw new LLMValidationError(`Invalid deduced card: ${card}`, {
        provider: 'unknown',
        agentName,
        taskType: 'memory_update',
        context: { card, validCards: allCards },
      });
    }
  }

  // Validate reasoning is a string
  if (typeof reasoning !== 'string') {
    throw new LLMValidationError('reasoning must be a string', {
      provider: 'unknown',
      agentName,
      taskType: 'memory_update',
      context: { reasoning },
    });
  }

  // Validate memorySummary is a string
  if (typeof memorySummary !== 'string') {
    throw new LLMValidationError('memorySummary must be a string', {
      provider: 'unknown',
      agentName,
      taskType: 'memory_update',
      context: { memorySummary },
    });
  }

  return true;
}

/**
 * Validates a challenge response from LLM.
 *
 * @param {Object} challenge - Challenge response to validate
 * @param {string} challenge.cardToShow - Card to show
 * @param {string} challenge.reasoning - Reasoning for choice
 * @param {Object} options - Validation options
 * @param {Array<string>} [options.availableCards] - Cards agent can show
 * @param {string} [options.agentName] - Agent name
 * @returns {boolean} True if valid
 * @throws {LLMValidationError} If challenge is invalid
 */
export function validateChallenge(challenge, { availableCards, agentName } = {}) {
  if (!challenge || typeof challenge !== 'object') {
    throw new LLMValidationError('Challenge must be an object', {
      provider: 'unknown',
      agentName,
      taskType: 'challenge',
      context: { challenge },
    });
  }

  const { cardToShow, reasoning } = challenge;

  // Validate cardToShow exists and is a string
  if (!cardToShow || typeof cardToShow !== 'string') {
    throw new LLMValidationError('cardToShow must be a non-empty string', {
      provider: 'unknown',
      agentName,
      taskType: 'challenge',
      context: { cardToShow },
    });
  }

  // Validate card is in available cards if provided
  if (availableCards && !availableCards.includes(cardToShow)) {
    throw new LLMValidationError(`Agent cannot show card ${cardToShow} (not in hand)`, {
      provider: 'unknown',
      agentName,
      taskType: 'challenge',
      context: { cardToShow, availableCards },
    });
  }

  // Validate card is a valid game card
  validateCard(cardToShow, { context: `challenge.cardToShow for ${agentName}` });

  // Validate reasoning is a string
  if (typeof reasoning !== 'string') {
    throw new LLMValidationError('reasoning must be a string', {
      provider: 'unknown',
      agentName,
      taskType: 'challenge',
      context: { reasoning },
    });
  }

  return true;
}

/**
 * Sanitizes a string by removing potentially dangerous characters.
 *
 * @param {string} input - String to sanitize
 * @param {Object} options - Sanitization options
 * @param {number} [options.maxLength=1000] - Maximum length
 * @returns {string} Sanitized string
 */
export function sanitizeString(input, { maxLength = 1000 } = {}) {
  if (typeof input !== 'string') {
    return '';
  }

  // Trim whitespace
  let sanitized = input.trim();

  // Limit length
  if (sanitized.length > maxLength) {
    sanitized = sanitized.substring(0, maxLength);
  }

  return sanitized;
}

/**
 * Validates that an object matches expected structure.
 *
 * @param {Object} obj - Object to validate
 * @param {Array<string>} requiredKeys - Required keys
 * @param {string} objectName - Name of object for error messages
 * @throws {GameValidationError} If validation fails
 * @returns {boolean} True if valid
 */
export function validateObjectStructure(obj, requiredKeys, objectName = 'object') {
  if (!obj || typeof obj !== 'object') {
    throw new GameValidationError(`${objectName} must be an object`, {
      context: { obj, requiredKeys },
    });
  }

  const missingKeys = requiredKeys.filter(key => !(key in obj));

  if (missingKeys.length > 0) {
    throw new GameValidationError(`${objectName} missing required keys: ${missingKeys.join(', ')}`, {
      context: { obj, requiredKeys, missingKeys },
    });
  }

  return true;
}
