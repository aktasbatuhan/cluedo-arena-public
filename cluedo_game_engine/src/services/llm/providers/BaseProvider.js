import { logger } from '../../../utils/logger.js';
import { LLMError, LLMTimeoutError } from '../../../utils/errors.js';

/**
 * Base class for LLM providers.
 * All provider implementations should extend this class.
 *
 * @abstract
 */
export class BaseProvider {
  /**
   * @param {string} name - Provider name
   * @param {Object} client - Provider-specific client instance
   * @param {Object} config - Provider configuration
   */
  constructor(name, client, config = {}) {
    if (new.target === BaseProvider) {
      throw new Error('BaseProvider is abstract and cannot be instantiated directly');
    }

    this.name = name;
    this.client = client;
    this.config = config;
  }

  /**
   * Calls the LLM with a prompt and returns the response text.
   *
   * @abstract
   * @param {string} prompt - The prompt to send to the LLM
   * @param {Object} options - Provider-specific options
   * @param {string} [options.model] - Model to use
   * @param {number} [options.temperature] - Sampling temperature
   * @param {string} [options.taskType] - Type of task (for logging)
   * @returns {Promise<string>} Response text from the LLM
   * @throws {LLMError} If the API call fails
   */
  async call(prompt, options = {}) {
    throw new Error('call() must be implemented by provider subclass');
  }

  /**
   * Checks if the provider supports a given model.
   *
   * @param {string} model - Model identifier
   * @returns {boolean}
   */
  supportsModel(model) {
    return false;
  }

  /**
   * Gets the provider name.
   * @returns {string}
   */
  getName() {
    return this.name;
  }

  /**
   * Validates that the client is initialized.
   * @throws {LLMError} If client is not initialized
   * @protected
   */
  _validateClient() {
    if (!this.client) {
      throw new LLMError(`${this.name} client not initialized`, {
        provider: this.name,
        statusCode: 500,
      });
    }
  }

  /**
   * Logs provider activity.
   * @param {string} level - Log level
   * @param {string} message - Log message
   * @param {Object} metadata - Additional metadata
   * @protected
   */
  _log(level, message, metadata = {}) {
    logger.log(level, `[${this.name.toUpperCase()}] ${message}`, {
      provider: this.name,
      ...metadata,
    });
  }

  /**
   * Handles provider-specific errors.
   * @param {Error} error - Original error
   * @param {Object} context - Error context
   * @returns {LLMError}
   * @protected
   */
  _handleError(error, context = {}) {
    if (error instanceof LLMError) {
      return error;
    }

    const errorContext = {
      provider: this.name,
      ...context,
    };

    // Check for timeout
    if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      return new LLMTimeoutError(error.message, errorContext);
    }

    // Check for HTTP errors
    if (error.response) {
      return new LLMError(error.response.data?.error?.message || error.message, {
        ...errorContext,
        statusCode: error.response.status,
      });
    }

    // Generic error
    return new LLMError(error.message, errorContext);
  }
}
