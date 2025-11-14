import config from '../../config/appConfig.js';
import { CohereProvider } from './providers/CohereProvider.js';
import { OpenRouterProvider } from './providers/OpenRouterProvider.js';
import { PredibaseProvider } from './providers/PredibaseProvider.js';
import { logger } from '../../utils/logger.js';
import { ConfigurationError } from '../../utils/errors.js';

/**
 * Factory for creating and managing LLM provider instances.
 * Handles provider initialization, caching, and selection.
 */
export class ProviderFactory {
  constructor() {
    this.providers = new Map();
    this.defaultProvider = null;
  }

  /**
   * Initializes all configured providers.
   * @returns {ProviderFactory} This factory instance
   */
  initialize() {
    const apiKeys = config.llm.apiKeys;

    // Initialize Cohere provider if API key is available
    if (apiKeys.cohere) {
      try {
        const cohereProvider = CohereProvider.initialize(apiKeys.cohere);
        this.providers.set('cohere', cohereProvider);
        logger.info('Cohere provider initialized');
      } catch (error) {
        logger.warn(`Failed to initialize Cohere provider: ${error.message}`);
      }
    }

    // Initialize OpenRouter provider if API key is available
    if (apiKeys.openRouter) {
      try {
        const openRouterProvider = OpenRouterProvider.initialize(apiKeys.openRouter);
        this.providers.set('openrouter', openRouterProvider);
        logger.info('OpenRouter provider initialized');
      } catch (error) {
        logger.warn(`Failed to initialize OpenRouter provider: ${error.message}`);
      }
    }

    // Initialize Predibase provider if API key is available
    if (apiKeys.predibase) {
      try {
        const predibaseProvider = PredibaseProvider.initialize(apiKeys.predibase);
        this.providers.set('predibase', predibaseProvider);
        logger.info('Predibase provider initialized (stub)');
      } catch (error) {
        logger.warn(`Failed to initialize Predibase provider: ${error.message}`);
      }
    }

    // Set default provider based on configuration
    this._setDefaultProvider(config.llm.backend.toLowerCase());

    // Ensure at least one provider is available
    if (this.providers.size === 0) {
      throw new ConfigurationError(
        'No LLM providers could be initialized. Check your API keys in .env'
      );
    }

    logger.info(`ProviderFactory initialized with ${this.providers.size} provider(s)`);
    return this;
  }

  /**
   * Sets the default provider.
   * @param {string} providerName - Name of the provider
   * @private
   */
  _setDefaultProvider(providerName) {
    const provider = this.providers.get(providerName);

    if (!provider) {
      logger.warn(
        `Requested default provider '${providerName}' not available, using first available`
      );
      this.defaultProvider = this.providers.values().next().value;
    } else {
      this.defaultProvider = provider;
    }

    logger.info(`Default provider set to: ${this.defaultProvider.getName()}`);
  }

  /**
   * Gets a provider by name.
   *
   * @param {string} providerName - Provider name
   * @returns {BaseProvider|null}
   */
  getProvider(providerName) {
    return this.providers.get(providerName.toLowerCase()) || null;
  }

  /**
   * Gets the appropriate provider for a given model.
   *
   * @param {string} model - Model identifier
   * @returns {BaseProvider}
   * @throws {ConfigurationError} If no suitable provider is found
   */
  getProviderForModel(model) {
    if (!model) {
      return this.defaultProvider;
    }

    // Check each provider to see if it supports the model
    for (const provider of this.providers.values()) {
      if (provider.supportsModel(model)) {
        logger.debug(`Selected provider ${provider.getName()} for model ${model}`);
        return provider;
      }
    }

    // Fall back to default provider
    logger.warn(
      `No provider found for model ${model}, using default: ${this.defaultProvider.getName()}`
    );
    return this.defaultProvider;
  }

  /**
   * Gets the default provider.
   * @returns {BaseProvider}
   */
  getDefaultProvider() {
    return this.defaultProvider;
  }

  /**
   * Lists all available provider names.
   * @returns {Array<string>}
   */
  listProviders() {
    return Array.from(this.providers.keys());
  }

  /**
   * Checks if a provider is available.
   * @param {string} providerName - Provider name
   * @returns {boolean}
   */
  hasProvider(providerName) {
    return this.providers.has(providerName.toLowerCase());
  }
}

// Export singleton instance
export const providerFactory = new ProviderFactory();
