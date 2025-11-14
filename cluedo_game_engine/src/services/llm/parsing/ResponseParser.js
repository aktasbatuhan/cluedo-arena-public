import Ajv from 'ajv';
import yaml from 'js-yaml';
import { logger } from '../../../utils/logger.js';
import { LLMValidationError } from '../../../utils/errors.js';
import { SUSPECTS, WEAPONS, ROOMS } from '../../../config/gameConstants.js';

/**
 * Parses and validates LLM responses.
 * Handles YAML parsing and schema validation.
 */
export class ResponseParser {
  constructor() {
    this.ajv = new Ajv();

    // Define JSON schemas for validation
    this.schemas = {
      suggestion: {
        type: 'object',
        properties: {
          suspect: { type: 'string', enum: SUSPECTS },
          weapon: { type: 'string', enum: WEAPONS },
          room: { type: 'string', enum: ROOMS },
          reasoning: { type: 'string', minLength: 1 },
        },
        required: ['suspect', 'weapon', 'room', 'reasoning'],
        additionalProperties: false,
      },

      accusation: {
        type: 'object',
        properties: {
          shouldAccuse: { type: 'boolean' },
          accusation: {
            type: 'object',
            properties: {
              suspect: { type: ['string', 'null'], enum: [...SUSPECTS, null] },
              weapon: { type: ['string', 'null'], enum: [...WEAPONS, null] },
              room: { type: ['string', 'null'], enum: [...ROOMS, null] },
            },
            required: ['suspect', 'weapon', 'room'],
          },
          reasoning: { type: 'string' },
        },
        required: ['shouldAccuse', 'accusation', 'reasoning'],
        additionalProperties: false,
      },

      memoryUpdate: {
        type: 'object',
        properties: {
          newlyDeducedCards: { type: 'array', items: { type: 'string' } },
          reasoning: { type: 'string' },
          memorySummary: { type: 'string' },
        },
        required: ['newlyDeducedCards', 'reasoning', 'memorySummary'],
        additionalProperties: false,
      },

      challenge: {
        type: 'object',
        properties: {
          cardToShow: { type: 'string' },
          reasoning: { type: 'string' },
        },
        required: ['cardToShow', 'reasoning'],
        additionalProperties: false,
      },
    };

    // Compile validators
    this.validators = {};
    for (const [name, schema] of Object.entries(this.schemas)) {
      this.validators[name] = this.ajv.compile(schema);
    }
  }

  /**
   * Extracts YAML from a response string, handling markdown code blocks.
   *
   * @param {string} response - Raw response text
   * @returns {Object|null} Parsed YAML object or null
   * @private
   */
  _extractYAML(response) {
    if (!response || typeof response !== 'string') {
      return null;
    }

    try {
      // Check for markdown fences and extract content if present
      const yamlMatch = response.match(/```(?:yaml)?\n?([\s\S]*?)\n?```/);
      const yamlContent = yamlMatch ? yamlMatch[1] : response;

      const parsed = yaml.load(yamlContent.trim());

      if (parsed !== null && typeof parsed === 'object') {
        return parsed;
      }

      logger.warn(`YAML parsing resulted in non-object type: ${typeof parsed}`);
      return null;
    } catch (error) {
      logger.error(`Failed to parse YAML: ${error.message}`, { response });
      return null;
    }
  }

  /**
   * Parses and validates a response against a schema.
   *
   * @param {string} response - Raw response text
   * @param {string} schemaType - Type of schema to validate against
   * @returns {{valid: boolean, data: Object|null, error: string|null}}
   */
  parse(response, schemaType) {
    const validator = this.validators[schemaType];

    if (!validator) {
      throw new Error(`Unknown schema type: ${schemaType}`);
    }

    // Extract YAML
    const parsed = this._extractYAML(response);

    if (!parsed) {
      return {
        valid: false,
        data: null,
        error: 'Failed to parse YAML response',
      };
    }

    // Validate against schema
    const isValid = validator(parsed);

    if (!isValid) {
      return {
        valid: false,
        data: parsed, // Return parsed data even if invalid
        error: this.ajv.errorsText(validator.errors),
      };
    }

    return {
      valid: true,
      data: parsed,
      error: null,
    };
  }

  /**
   * Parses and validates a suggestion response.
   *
   * @param {string} response - Raw response text
   * @param {Object} context - Context for error messages
   * @returns {Object} Parsed suggestion
   * @throws {LLMValidationError} If validation fails
   */
  parseSuggestion(response, context = {}) {
    const result = this.parse(response, 'suggestion');

    if (!result.valid) {
      throw new LLMValidationError('Invalid suggestion response', {
        response: result.data,
        validationErrors: [result.error],
        ...context,
      });
    }

    return result.data;
  }

  /**
   * Parses and validates an accusation response.
   *
   * @param {string} response - Raw response text
   * @param {Object} context - Context for error messages
   * @returns {Object} Parsed accusation
   * @throws {LLMValidationError} If validation fails
   */
  parseAccusation(response, context = {}) {
    const result = this.parse(response, 'accusation');

    if (!result.valid) {
      throw new LLMValidationError('Invalid accusation response', {
        response: result.data,
        validationErrors: [result.error],
        ...context,
      });
    }

    return result.data;
  }

  /**
   * Parses and validates a memory update response.
   *
   * @param {string} response - Raw response text
   * @param {Object} context - Context for error messages
   * @returns {Object} Parsed memory update
   * @throws {LLMValidationError} If validation fails
   */
  parseMemoryUpdate(response, context = {}) {
    const result = this.parse(response, 'memoryUpdate');

    if (!result.valid) {
      throw new LLMValidationError('Invalid memory update response', {
        response: result.data,
        validationErrors: [result.error],
        ...context,
      });
    }

    return result.data;
  }

  /**
   * Parses and validates a challenge response.
   *
   * @param {string} response - Raw response text
   * @param {Object} context - Context for error messages
   * @returns {Object} Parsed challenge
   * @throws {LLMValidationError} If validation fails
   */
  parseChallenge(response, context = {}) {
    const result = this.parse(response, 'challenge');

    if (!result.valid) {
      throw new LLMValidationError('Invalid challenge response', {
        response: result.data,
        validationErrors: [result.error],
        ...context,
      });
    }

    return result.data;
  }

  /**
   * Parses response with fallback on validation failure.
   *
   * @param {string} response - Raw response text
   * @param {string} schemaType - Schema type
   * @param {Function} fallbackFn - Function to generate fallback value
   * @returns {Object}
   */
  parseWithFallback(response, schemaType, fallbackFn) {
    try {
      const methodName = `parse${schemaType.charAt(0).toUpperCase()}${schemaType.slice(1)}`;
      return this[methodName](response);
    } catch (error) {
      logger.warn(`Response parsing failed, using fallback: ${error.message}`);
      return fallbackFn(error);
    }
  }
}

// Export singleton instance
export const responseParser = new ResponseParser();
