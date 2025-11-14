/**
 * Custom error classes for better error handling and debugging.
 * @module utils/errors
 */

/**
 * Base error class for all application errors.
 * Extends the native Error class with additional context.
 */
export class AppError extends Error {
  /**
   * @param {string} message - Error message
   * @param {Object} options - Additional error options
   * @param {number} [options.statusCode=500] - HTTP status code
   * @param {boolean} [options.isOperational=true] - Whether error is operational (expected)
   * @param {Object} [options.context={}] - Additional context data
   */
  constructor(message, { statusCode = 500, isOperational = true, context = {} } = {}) {
    super(message);
    this.name = this.constructor.name;
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.context = context;
    this.timestamp = new Date().toISOString();

    Error.captureStackTrace(this, this.constructor);
  }

  /**
   * Returns a JSON representation of the error.
   * @returns {Object}
   */
  toJSON() {
    return {
      name: this.name,
      message: this.message,
      statusCode: this.statusCode,
      isOperational: this.isOperational,
      context: this.context,
      timestamp: this.timestamp,
      stack: this.stack,
    };
  }
}

/**
 * Error thrown when LLM API calls fail.
 */
export class LLMError extends AppError {
  /**
   * @param {string} message - Error message
   * @param {Object} options - Additional error options
   * @param {string} [options.provider] - LLM provider name
   * @param {string} [options.model] - Model name
   * @param {string} [options.taskType] - Type of LLM task
   * @param {number} [options.attempt] - Attempt number (for retries)
   */
  constructor(
    message,
    { provider, model, taskType, attempt, statusCode = 500, context = {} } = {}
  ) {
    super(message, {
      statusCode,
      context: {
        ...context,
        provider,
        model,
        taskType,
        attempt,
      },
    });
  }
}

/**
 * Error thrown when LLM API calls timeout.
 */
export class LLMTimeoutError extends LLMError {
  constructor(message, options = {}) {
    super(message, { ...options, statusCode: 504 });
  }
}

/**
 * Error thrown when LLM response validation fails.
 */
export class LLMValidationError extends LLMError {
  /**
   * @param {string} message - Error message
   * @param {Object} options - Additional error options
   * @param {Object} [options.response] - Invalid response object
   * @param {Object} [options.schema] - Expected schema
   * @param {Array} [options.validationErrors] - Validation error details
   */
  constructor(message, { response, schema, validationErrors, ...options } = {}) {
    super(message, {
      ...options,
      statusCode: 422,
      context: {
        ...options.context,
        response,
        schema,
        validationErrors,
      },
    });
  }
}

/**
 * Error thrown when configuration is invalid.
 */
export class ConfigurationError extends AppError {
  constructor(message, options = {}) {
    super(message, { ...options, statusCode: 500, isOperational: false });
  }
}

/**
 * Error thrown when game validation fails.
 */
export class GameValidationError extends AppError {
  /**
   * @param {string} message - Error message
   * @param {Object} options - Additional error options
   * @param {string} [options.agentName] - Agent name
   * @param {number} [options.turnNumber] - Turn number
   */
  constructor(message, { agentName, turnNumber, ...options } = {}) {
    super(message, {
      ...options,
      statusCode: 400,
      context: {
        ...options.context,
        agentName,
        turnNumber,
      },
    });
  }
}

/**
 * Error thrown when network operations fail.
 */
export class NetworkError extends AppError {
  /**
   * @param {string} message - Error message
   * @param {Object} options - Additional error options
   * @param {string} [options.url] - Request URL
   * @param {string} [options.method] - HTTP method
   * @param {Error} [options.originalError] - Original error object
   */
  constructor(message, { url, method, originalError, ...options } = {}) {
    super(message, {
      ...options,
      statusCode: 503,
      context: {
        ...options.context,
        url,
        method,
        originalError: originalError?.message,
      },
    });
  }
}

/**
 * Determines if an error is retryable.
 * @param {Error} error - Error to check
 * @returns {boolean}
 */
export function isRetryableError(error) {
  // Network errors are retryable
  if (error instanceof NetworkError) {
    return true;
  }

  // Timeout errors are retryable
  if (error instanceof LLMTimeoutError) {
    return true;
  }

  // Some LLM errors are retryable (e.g., rate limits, temporary server errors)
  if (error instanceof LLMError) {
    const retryableStatusCodes = [429, 500, 502, 503, 504];
    return retryableStatusCodes.includes(error.statusCode);
  }

  // Check for common retryable error patterns
  if (error.code) {
    const retryableCodes = ['ECONNRESET', 'ETIMEDOUT', 'ENOTFOUND', 'ECONNREFUSED'];
    return retryableCodes.includes(error.code);
  }

  return false;
}

/**
 * Creates an appropriate error from an axios error.
 * @param {Error} axiosError - Axios error object
 * @param {Object} context - Additional context
 * @returns {AppError}
 */
export function fromAxiosError(axiosError, context = {}) {
  if (axiosError.code === 'ECONNABORTED') {
    return new LLMTimeoutError('Request timed out', context);
  }

  if (axiosError.response) {
    // Server responded with error status
    return new LLMError(axiosError.response.data?.error?.message || axiosError.message, {
      ...context,
      statusCode: axiosError.response.status,
    });
  }

  if (axiosError.request) {
    // Request made but no response received
    return new NetworkError('No response received from server', {
      ...context,
      originalError: axiosError,
    });
  }

  // Something else happened
  return new AppError(axiosError.message, {
    statusCode: 500,
    context,
  });
}
