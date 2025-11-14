/**
 * LLM Service - Orchestrates LLM interactions for Cluedo AI agents.
 *
 * This refactored service acts as a thin orchestration layer that:
 * - Routes requests to appropriate providers (Cohere, OpenRouter, etc.)
 * - Builds prompts using PromptBuilder
 * - Parses and validates responses using ResponseParser
 * - Handles errors with retry logic
 * - Logs all interactions for analysis
 *
 * @module services/llm
 */

import { logger } from '../utils/logger.js';
import { providerFactory } from './llm/ProviderFactory.js';
import { PromptBuilder } from './llm/prompts/PromptBuilder.js';
import { responseParser } from './llm/parsing/ResponseParser.js';
import { LoggingService } from './LoggingService.js';
import { withRetryAndTimeout } from '../utils/retry.js';
import { LLMError, LLMValidationError } from '../utils/errors.js';
import config from '../config/appConfig.js';

/**
 * List of LLM models to be used for AI agents.
 * Each game will randomly assign these models to agents.
 *
 * @type {Array<string>}
 */
export const MODEL_LIST = [
  'anthropic/claude-3.5-sonnet',
  'cohere/command-a',
  'cohere/command-r-plus',
  'google/gemini-2.5-flash-preview',
  'openai/gpt-4o-2024-11-20',
];

/**
 * Main LLM service class.
 * Provides high-level methods for different LLM tasks.
 */
export class LLMService {
  /**
   * Initializes the LLM service.
   * Must be called before using any LLM methods.
   */
  static initialize() {
    providerFactory.initialize();
    logger.info('LLMService initialized');
  }

  /**
   * Calls an LLM with retry and timeout logic.
   *
   * @private
   * @param {BaseProvider} provider - Provider to use
   * @param {string} prompt - Prompt to send
   * @param {Object} options - Call options
   * @returns {Promise<string>} Response text
   */
  static async _callWithRetry(provider, prompt, options = {}) {
    return withRetryAndTimeout(
      async () => provider.call(prompt, options),
      {
        timeoutMs: config.llm.requestTimeout,
        retryOptions: {
          ...config.llm.retry,
          operationName: `LLM call (${provider.getName()})`,
        },
      }
    );
  }

  /**
   * Logs an LLM interaction for analysis.
   *
   * @private
   * @param {Object} payload - Logging payload
   */
  static async _logInteraction(payload) {
    try {
      await LoggingService.logLLMInteraction(payload);
    } catch (error) {
      logger.error(`Failed to log LLM interaction: ${error.message}`);
    }
  }

  /**
   * Makes a strategic suggestion for an agent.
   *
   * @param {Object} agent - The agent making the suggestion
   * @param {Object} gameState - Current game state
   * @returns {Promise<Object>} Suggestion with suspect, weapon, room, and reasoning
   */
  static async makeSuggestion(agent, gameState) {
    const startTime = Date.now();
    const taskType = 'suggestion';

    const loggingPayload = {
      type: taskType,
      agent: agent.name,
      model: agent.model,
      input: {},
      output: null,
      error: null,
      parsedOutput: null,
      validationStatus: 'pending',
    };

    try {
      // Get provider for model
      const provider = providerFactory.getProviderForModel(agent.model);

      // Build prompt
      const memoryState = await agent.memory.formatMemoryForLLM();
      const prompt = PromptBuilder.buildSuggestionPrompt(agent, gameState, memoryState);
      loggingPayload.input = { prompt, gameState };

      // Call LLM with retry
      const responseText = await LLMService._callWithRetry(provider, prompt, {
        model: agent.model,
        taskType,
      });
      loggingPayload.output = responseText;

      // Parse and validate response
      const parsedSuggestion = responseParser.parseSuggestion(responseText, {
        provider: provider.getName(),
        model: agent.model,
        agentName: agent.name,
      });
      loggingPayload.parsedOutput = parsedSuggestion;

      // Validate room matches agent location
      if (parsedSuggestion.room !== agent.location) {
        logger.warn(
          `${agent.name}: LLM suggested room (${parsedSuggestion.room}) doesn't match location (${agent.location}). Correcting.`
        );
        parsedSuggestion.room = agent.location;
        parsedSuggestion.reasoning += ` (Corrected room to agent's location: ${agent.location})`;
        loggingPayload.validationStatus = 'corrected_room';
      } else {
        loggingPayload.validationStatus = 'passed';
      }

      // Log interaction
      await LLMService._logInteraction(loggingPayload);

      logger.info(`Suggestion for ${agent.name} took ${Date.now() - startTime}ms`);

      return parsedSuggestion;
    } catch (error) {
      logger.error(`${agent.name} failed ${taskType}: ${error.message}`, { error });

      loggingPayload.error = error.message;
      loggingPayload.validationStatus =
        error instanceof LLMValidationError ? 'failed_validation' : 'failed_api_call';

      await LLMService._logInteraction(loggingPayload);

      // Return fallback suggestion
      const fallbackSuspect =
        gameState.availableSuspects?.find(s => !agent.memory.eliminatedCards.has(s)) ||
        gameState.availableSuspects?.[0] ||
        'Miss Scarlet';
      const fallbackWeapon =
        gameState.availableWeapons?.find(w => !agent.memory.eliminatedCards.has(w)) ||
        gameState.availableWeapons?.[0] ||
        'Candlestick';

      return {
        suspect: fallbackSuspect,
        weapon: fallbackWeapon,
        room: agent.location || 'Lounge',
        reasoning: `(Fallback: ${error.message})`,
        error: error.message,
      };
    }
  }

  /**
   * Updates an agent's memory based on turn events.
   *
   * @param {Object} agent - The agent whose memory is being updated
   * @param {Object} memory - Agent's memory object
   * @param {Array<string>} turnEvents - Events from the turn
   * @returns {Promise<Object>} Deduced cards and summary
   */
  static async updateMemory(agent, memory, turnEvents) {
    const startTime = Date.now();
    const taskType = 'memory_update';

    const loggingPayload = {
      type: taskType,
      agent: agent.name,
      model: agent.model,
      input: {},
      output: null,
      error: null,
      parsedOutput: null,
      validationStatus: 'pending',
    };

    // Skip if no events
    if (!turnEvents || turnEvents.length === 0) {
      return {
        deducedCards: [],
        summary: '(Memory update skipped, no events)',
      };
    }

    try {
      // Get provider
      const provider = providerFactory.getProviderForModel(agent.model);

      // Build prompt
      const prompt = PromptBuilder.buildMemoryUpdatePrompt(
        agent,
        memory,
        turnEvents,
        agent.game.currentTurn
      );
      loggingPayload.input = { prompt, turnEvents };

      // Call LLM
      const responseText = await LLMService._callWithRetry(provider, prompt, {
        model: agent.model,
        taskType,
      });
      loggingPayload.output = responseText;

      // Parse response
      const parsedResult = responseParser.parseMemoryUpdate(responseText, {
        provider: provider.getName(),
        model: agent.model,
        agentName: agent.name,
      });
      loggingPayload.parsedOutput = parsedResult;
      loggingPayload.validationStatus = 'passed';

      const deducedCards = parsedResult.newlyDeducedCards || [];
      const summary =
        parsedResult.memorySummary ||
        parsedResult.reasoning ||
        '(No summary provided by LLM)';
      const reasoning = parsedResult.reasoning || '(No reasoning provided by LLM)';

      // Update memory
      if (memory.update) {
        await memory.update(summary, deducedCards, reasoning);
      }

      // Log interaction
      await LLMService._logInteraction(loggingPayload);

      logger.info(`Memory update for ${agent.name} took ${Date.now() - startTime}ms`);

      return { deducedCards, summary };
    } catch (error) {
      logger.error(`${agent.name} failed ${taskType}: ${error.message}`, { error });

      loggingPayload.error = error.message;
      loggingPayload.validationStatus =
        error instanceof LLMValidationError ? 'failed_validation' : 'failed_api_call';

      await LLMService._logInteraction(loggingPayload);

      // Return fallback
      return {
        deducedCards: [],
        summary: `(Fallback: ${error.message})`,
        error: error.message,
      };
    }
  }

  /**
   * Evaluates a challenge and decides which card to show.
   *
   * @param {Object} agent - The agent responding to the challenge
   * @param {Object} suggestion - The suggestion being challenged
   * @param {Array<string>} matchingCards - Cards the agent can show
   * @returns {Promise<Object>} Card to show and reasoning
   */
  static async evaluateChallenge(agent, suggestion, matchingCards) {
    const startTime = Date.now();
    const taskType = 'evaluate_challenge';

    const loggingPayload = {
      type: taskType,
      agent: agent.name,
      model: agent.model,
      input: {},
      output: null,
      error: null,
      parsedOutput: null,
      validationStatus: 'pending',
    };

    // No need to call LLM if no matching cards
    if (!matchingCards || matchingCards.length === 0) {
      return {
        cardToShow: null,
        reasoning: 'No matching cards to show',
      };
    }

    try {
      // Get provider
      const provider = providerFactory.getProviderForModel(agent.model);

      // Build prompt
      const memoryState = await agent.memory.formatMemoryForLLM();
      const prompt = PromptBuilder.buildChallengePrompt(
        agent,
        suggestion,
        matchingCards,
        memoryState
      );
      loggingPayload.input = { prompt, suggestion, matchingCards };

      // Call LLM
      const responseText = await LLMService._callWithRetry(provider, prompt, {
        model: agent.model,
        taskType,
      });
      loggingPayload.output = responseText;

      // Parse response
      const parsedResult = responseParser.parseChallenge(responseText, {
        provider: provider.getName(),
        model: agent.model,
        agentName: agent.name,
      });
      loggingPayload.parsedOutput = parsedResult;

      const cardToShow = parsedResult.cardToShow;
      let reasoning = parsedResult.reasoning || '(No reasoning provided by LLM)';

      // Validate chosen card
      if (!matchingCards.includes(cardToShow)) {
        logger.warn(
          `${agent.name}: LLM chose invalid card (${cardToShow}). Not in matching set. Using fallback.`
        );
        const fallbackCard = matchingCards[0];
        reasoning = `(Fallback: LLM chose invalid card ${cardToShow}). ${reasoning}`;
        loggingPayload.validationStatus = 'corrected_invalid_card';
        loggingPayload.parsedOutput = { cardToShow: fallbackCard, reasoning };

        await LLMService._logInteraction(loggingPayload);

        return { cardToShow: fallbackCard, reasoning };
      }

      loggingPayload.validationStatus = 'passed';
      await LLMService._logInteraction(loggingPayload);

      logger.info(`Challenge evaluation for ${agent.name} took ${Date.now() - startTime}ms`);

      return { cardToShow, reasoning };
    } catch (error) {
      logger.error(`${agent.name} failed ${taskType}: ${error.message}`, { error });

      loggingPayload.error = error.message;
      loggingPayload.validationStatus =
        error instanceof LLMValidationError ? 'failed_validation' : 'failed_api_call';

      await LLMService._logInteraction(loggingPayload);

      // Return fallback
      const fallbackCard = matchingCards[0] || null;
      return {
        cardToShow: fallbackCard,
        reasoning: `(Fallback: ${error.message}). Showing ${fallbackCard || 'nothing'}.`,
        error: error.message,
      };
    }
  }

  /**
   * Decides whether an agent should make an accusation.
   *
   * @param {Object} agent - The agent considering the accusation
   * @param {Object} gameState - Current game state
   * @returns {Promise<Object>} Accusation decision
   */
  static async considerAccusation(agent, gameState) {
    const startTime = Date.now();
    const taskType = 'consider_accusation';

    const loggingPayload = {
      type: taskType,
      agent: agent.name,
      model: agent.model,
      input: {},
      output: null,
      error: null,
      parsedOutput: null,
      validationStatus: 'pending',
    };

    try {
      // Get provider
      const provider = providerFactory.getProviderForModel(agent.model);

      // Build prompt
      const memoryState = await agent.memory.formatMemoryForLLM();
      const prompt = PromptBuilder.buildAccusationPrompt(agent, gameState, memoryState);
      loggingPayload.input = { prompt, gameState };

      // Call LLM
      const responseText = await LLMService._callWithRetry(provider, prompt, {
        model: agent.model,
        taskType,
      });
      loggingPayload.output = responseText;

      // Parse response
      const parsedResult = responseParser.parseAccusation(responseText, {
        provider: provider.getName(),
        model: agent.model,
        agentName: agent.name,
      });
      loggingPayload.parsedOutput = parsedResult;

      // Validate logic: if accusing, accusation details must be complete
      if (
        parsedResult.shouldAccuse &&
        (!parsedResult.accusation ||
          !parsedResult.accusation.suspect ||
          !parsedResult.accusation.weapon ||
          !parsedResult.accusation.room)
      ) {
        logger.warn(
          `${agent.name}: shouldAccuse is true but accusation details missing. Overriding to false.`
        );
        parsedResult.shouldAccuse = false;
        if (!parsedResult.accusation) parsedResult.accusation = {};
        parsedResult.accusation.suspect = null;
        parsedResult.accusation.weapon = null;
        parsedResult.accusation.room = null;
        parsedResult.reasoning +=
          ' (Invalid accusation details provided, overriding shouldAccuse to false)';
        loggingPayload.validationStatus = 'corrected_logic';
      } else {
        loggingPayload.validationStatus = 'passed';
      }

      await LLMService._logInteraction(loggingPayload);

      logger.info(`Accusation consideration for ${agent.name} took ${Date.now() - startTime}ms`);

      return parsedResult;
    } catch (error) {
      logger.error(`${agent.name} failed ${taskType}: ${error.message}`, { error });

      loggingPayload.error = error.message;
      loggingPayload.validationStatus =
        error instanceof LLMValidationError ? 'failed_validation' : 'failed_api_call';

      await LLMService._logInteraction(loggingPayload);

      // Return safe fallback (don't accuse)
      return {
        shouldAccuse: false,
        accusation: { suspect: null, weapon: null, room: null },
        reasoning: `(Fallback: ${error.message}). Defaulting to not accuse.`,
        error: error.message,
      };
    }
  }

  /**
   * Sets the backend provider.
   * @deprecated Use ProviderFactory configuration instead
   * @param {string} backendName - Provider name
   */
  static setBackend(backendName) {
    logger.warn('setBackend() is deprecated. Providers are auto-selected based on model.');
    // For backwards compatibility, we can still update the default
    // But this is now handled by ProviderFactory
  }

  /**
   * Gets backend configuration for a model.
   * @deprecated Use ProviderFactory.getProviderForModel() instead
   * @param {string} model - Model identifier
   * @returns {Object}
   */
  static getBackendConfig(model) {
    logger.warn('getBackendConfig() is deprecated. Use ProviderFactory directly.');
    const provider = providerFactory.getProviderForModel(model);
    return {
      client: provider.client,
      type: provider.getName(),
      model: model,
    };
  }
}

// Initialize on module load
LLMService.initialize();
