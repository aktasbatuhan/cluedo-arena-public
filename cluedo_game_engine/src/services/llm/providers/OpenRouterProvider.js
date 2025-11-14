import { BaseProvider } from './BaseProvider.js';
import { LLMError } from '../../../utils/errors.js';
import { OpenAI } from 'openai';
import config from '../../../config/appConfig.js';

/**
 * OpenRouter LLM provider implementation.
 * Uses the OpenAI SDK with OpenRouter's API endpoint.
 */
export class OpenRouterProvider extends BaseProvider {
  /**
   * @param {string} apiKey - OpenRouter API key
   * @param {Object} providerConfig - Provider configuration
   */
  constructor(apiKey, providerConfig = {}) {
    const client = apiKey
      ? new OpenAI({
          baseURL: 'https://openrouter.ai/api/v1',
          apiKey: apiKey,
        })
      : null;

    super('openrouter', client, providerConfig);

    if (!apiKey) {
      this._log('warn', 'OpenRouter API key not provided');
    }
  }

  /**
   * Calls OpenRouter API with a prompt.
   *
   * @param {string} prompt - The prompt to send
   * @param {Object} options - Call options
   * @param {string} [options.model] - Model to use
   * @param {number} [options.temperature=0.1] - Sampling temperature
   * @param {string} [options.taskType=''] - Task type for logging
   * @returns {Promise<string>} Response text
   */
  async call(prompt, options = {}) {
    this._validateClient();

    const {
      model = 'openai/gpt-4o-mini',
      temperature = 0.1,
      taskType = 'unknown',
    } = options;

    this._log('debug', `Calling API for ${taskType}`, { model });

    try {
      const completion = await this.client.chat.completions.create({
        extra_headers: {
          'HTTP-Referer': config.site.url,
          'X-Title': config.site.name,
        },
        model,
        messages: [{ role: 'user', content: prompt }],
        temperature,
        // Note: response_format with JSON mode can be added if needed
        // response_format: { type: 'json_object' }
      });

      const responseText = completion.choices[0]?.message?.content;

      if (!responseText) {
        throw new LLMError('Empty response from OpenRouter API', {
          provider: this.name,
          model,
          taskType,
        });
      }

      this._log('debug', `Received response for ${taskType}`, {
        model,
        responseLength: responseText.length,
      });

      return responseText;
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
    // OpenRouter supports many models with prefixes like:
    // openai/, google/, anthropic/, mistralai/, meta-llama/, etc.
    const supportedPrefixes = [
      'openai/',
      'google/',
      'anthropic/',
      'mistralai/',
      'meta-llama/',
      'qwen/',
      'deepseek/',
    ];

    return supportedPrefixes.some(prefix => model?.startsWith(prefix));
  }

  /**
   * Initializes the OpenRouter provider with an API key.
   * @param {string} apiKey - OpenRouter API key
   * @returns {OpenRouterProvider}
   */
  static initialize(apiKey) {
    if (!apiKey) {
      throw new Error(
        'OPENROUTER_API_KEY environment variable is required for OpenRouter provider'
      );
    }
    return new OpenRouterProvider(apiKey);
  }
}
