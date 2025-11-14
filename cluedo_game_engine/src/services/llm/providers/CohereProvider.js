import { BaseProvider } from './BaseProvider.js';
import { LLMError } from '../../../utils/errors.js';
import { CohereClient } from 'cohere-ai';

/**
 * Cohere LLM provider implementation.
 */
export class CohereProvider extends BaseProvider {
  /**
   * @param {string} apiKey - Cohere API key
   * @param {Object} config - Provider configuration
   */
  constructor(apiKey, config = {}) {
    const client = apiKey ? new CohereClient({ token: apiKey }) : null;
    super('cohere', client, config);

    if (!apiKey) {
      this._log('warn', 'Cohere API key not provided');
    }
  }

  /**
   * Calls Cohere API with a prompt.
   *
   * @param {string} prompt - The prompt to send
   * @param {Object} options - Call options
   * @param {string} [options.model='command-r'] - Cohere model to use
   * @param {number} [options.temperature=0.1] - Sampling temperature
   * @param {string} [options.taskType=''] - Task type for logging
   * @returns {Promise<string>} Response text
   */
  async call(prompt, options = {}) {
    this._validateClient();

    const {
      model = 'command-r',
      temperature = 0.1,
      taskType = 'unknown',
    } = options;

    this._log('debug', `Calling API for ${taskType}`, { model });

    try {
      // Check if model supports JSON response format
      const supportsJsonResponseFormat = model.startsWith('command-r');

      const apiParams = {
        model,
        message: prompt,
        temperature,
      };

      // Add JSON response format if supported
      if (supportsJsonResponseFormat) {
        apiParams.response_format = { type: 'json_object' };
      }

      const response = await this.client.chat(apiParams);

      if (!response || !response.text) {
        throw new LLMError('Empty response from Cohere API', {
          provider: this.name,
          model,
          taskType,
        });
      }

      this._log('debug', `Received response for ${taskType}`, {
        model,
        responseLength: response.text.length,
      });

      return response.text;
    } catch (error) {
      throw this._handleError(error, {
        model,
        taskType,
      });
    }
  }

  /**
   * Checks if this provider supports a given model.
   * @param {string} model - Model identifier
   * @returns {boolean}
   */
  supportsModel(model) {
    // Cohere models typically start with 'command' or 'c4ai-aya'
    return (
      model?.startsWith('command') ||
      model?.startsWith('c4ai-aya') ||
      model === 'cohere/command-a' ||
      model === 'cohere/command-r-plus'
    );
  }

  /**
   * Initializes the Cohere client with an API key.
   * @param {string} apiKey - Cohere API key
   */
  static initialize(apiKey) {
    if (!apiKey) {
      throw new Error('CO_API_KEY environment variable is required for Cohere provider');
    }
    return new CohereProvider(apiKey);
  }
}
