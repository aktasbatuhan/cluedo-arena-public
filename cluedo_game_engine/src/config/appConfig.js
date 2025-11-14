import 'dotenv/config';
import { logger } from '../utils/logger.js';

/**
 * Centralized application configuration.
 * All configuration values and environment variables are accessed through this module.
 *
 * @module config/appConfig
 */

/**
 * Validates that required environment variables are set.
 * @throws {Error} If required environment variables are missing
 * @private
 */
function validateRequiredEnvVars() {
  const errors = [];

  // Get the LLM backend to determine which API keys are required
  const backend = process.env.LLM_BACKEND || 'OPENROUTER';

  // Check for required API keys based on backend
  if (backend === 'COHERE' && !process.env.COHERE_API_KEY) {
    errors.push('COHERE_API_KEY is required when using COHERE backend');
  }

  if (backend === 'OPENROUTER' && !process.env.OPENROUTER_API_KEY) {
    errors.push('OPENROUTER_API_KEY is required when using OPENROUTER backend');
  }

  if (backend === 'PREDIBASE' && !process.env.PREDIBASE_API_KEY) {
    errors.push('PREDIBASE_API_KEY is required when using PREDIBASE backend');
  }

  if (errors.length > 0) {
    throw new Error(`Environment validation failed:\n${errors.join('\n')}`);
  }
}

/**
 * Parses and validates configuration values.
 * @private
 */
function parseConfig() {
  // Validate required variables first
  validateRequiredEnvVars();

  const config = {
    // Server Configuration
    server: {
      port: parseInt(process.env.PORT, 10) || 3000,
      environment: process.env.NODE_ENV || 'development',
    },

    // Game Configuration
    game: {
      maxTurns: parseInt(process.env.MAX_TURNS, 10) || 120,
      numberOfAgents: 6,
      defaultGameMode: 'spectate',
    },

    // Database Configuration
    database: {
      mongoUri: process.env.MONGO_URI || null,
    },

    // Logging Configuration
    logging: {
      level: process.env.LOG_LEVEL || 'info',
    },

    // LLM Configuration
    llm: {
      backend: process.env.LLM_BACKEND || 'OPENROUTER',
      requestTimeout: parseInt(process.env.LLM_REQUEST_TIMEOUT, 10) || 60000,

      // API Keys
      apiKeys: {
        cohere: process.env.COHERE_API_KEY || null,
        openRouter: process.env.OPENROUTER_API_KEY || null,
        predibase: process.env.PREDIBASE_API_KEY || null,
      },

      // Provider URLs
      providers: {
        artWrapperUrl: process.env.ART_WRAPPER_URL || 'http://localhost:5001',
        openRouterBaseUrl: 'https://openrouter.ai/api/v1',
      },

      // Retry Configuration
      retry: {
        maxAttempts: 3,
        initialDelayMs: 1000,
        maxDelayMs: 10000,
        backoffMultiplier: 2,
      },
    },

    // Site Metadata
    site: {
      url: process.env.SITE_URL || 'http://localhost:3000',
      name: process.env.SITE_NAME || 'Cluedo AI Arena',
    },
  };

  // Validate numeric values
  if (config.server.port < 1 || config.server.port > 65535) {
    throw new Error(`Invalid PORT value: ${config.server.port}. Must be between 1 and 65535.`);
  }

  if (config.game.maxTurns < 1) {
    throw new Error(`Invalid MAX_TURNS value: ${config.game.maxTurns}. Must be at least 1.`);
  }

  if (config.llm.requestTimeout < 1000) {
    throw new Error(`Invalid LLM_REQUEST_TIMEOUT value: ${config.llm.requestTimeout}. Must be at least 1000ms.`);
  }

  // Validate LLM backend
  const validBackends = ['COHERE', 'OPENROUTER', 'PREDIBASE', 'ART'];
  if (!validBackends.includes(config.llm.backend)) {
    throw new Error(`Invalid LLM_BACKEND value: ${config.llm.backend}. Must be one of: ${validBackends.join(', ')}`);
  }

  return config;
}

/**
 * Application configuration object.
 * Frozen to prevent runtime modifications.
 */
let config;

try {
  config = parseConfig();
  Object.freeze(config);

  logger.info('Configuration loaded successfully', {
    backend: config.llm.backend,
    port: config.server.port,
    environment: config.server.environment,
  });
} catch (error) {
  logger.error('Configuration validation failed', { error: error.message });
  throw error;
}

export default config;
