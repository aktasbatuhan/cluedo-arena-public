import { logger } from './logger.js';
import { isRetryableError } from './errors.js';
import config from '../config/appConfig.js';

/**
 * Retry utility for handling transient failures with exponential backoff.
 * @module utils/retry
 */

/**
 * Sleeps for a specified duration.
 * @param {number} ms - Milliseconds to sleep
 * @returns {Promise<void>}
 * @private
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Calculates delay for exponential backoff with jitter.
 * @param {number} attempt - Current attempt number (0-indexed)
 * @param {number} baseDelay - Base delay in milliseconds
 * @param {number} maxDelay - Maximum delay in milliseconds
 * @param {number} multiplier - Backoff multiplier
 * @returns {number} Delay in milliseconds
 * @private
 */
function calculateBackoff(attempt, baseDelay, maxDelay, multiplier) {
  // Calculate exponential backoff
  const exponentialDelay = baseDelay * Math.pow(multiplier, attempt);

  // Cap at max delay
  const cappedDelay = Math.min(exponentialDelay, maxDelay);

  // Add jitter (random variation of ±25%)
  const jitter = cappedDelay * 0.25 * (Math.random() * 2 - 1);

  return Math.floor(cappedDelay + jitter);
}

/**
 * Executes a function with retry logic and exponential backoff.
 *
 * @template T
 * @param {Function} fn - Async function to execute
 * @param {Object} options - Retry options
 * @param {number} [options.maxAttempts] - Maximum number of attempts
 * @param {number} [options.initialDelayMs] - Initial delay in milliseconds
 * @param {number} [options.maxDelayMs] - Maximum delay in milliseconds
 * @param {number} [options.backoffMultiplier] - Backoff multiplier
 * @param {Function} [options.shouldRetry] - Custom function to determine if error is retryable
 * @param {Function} [options.onRetry] - Callback called before each retry
 * @param {string} [options.operationName] - Name of operation for logging
 * @returns {Promise<T>} Result of the function
 * @throws {Error} Last error if all retries fail
 *
 * @example
 * const result = await withRetry(
 *   async () => await fetchData(),
 *   {
 *     maxAttempts: 3,
 *     operationName: 'fetchData'
 *   }
 * );
 */
export async function withRetry(fn, options = {}) {
  const {
    maxAttempts = config.llm.retry.maxAttempts,
    initialDelayMs = config.llm.retry.initialDelayMs,
    maxDelayMs = config.llm.retry.maxDelayMs,
    backoffMultiplier = config.llm.retry.backoffMultiplier,
    shouldRetry = isRetryableError,
    onRetry = null,
    operationName = 'operation',
  } = options;

  let lastError;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      // Execute the function
      const result = await fn();

      // Log success if this was a retry
      if (attempt > 0) {
        logger.info(`${operationName} succeeded after ${attempt + 1} attempts`);
      }

      return result;
    } catch (error) {
      lastError = error;

      // Check if we should retry
      const isLastAttempt = attempt === maxAttempts - 1;
      const canRetry = shouldRetry(error);

      if (isLastAttempt || !canRetry) {
        // Log final failure
        logger.error(`${operationName} failed after ${attempt + 1} attempts`, {
          error: error.message,
          stack: error.stack,
          canRetry,
        });
        throw error;
      }

      // Calculate delay
      const delay = calculateBackoff(attempt, initialDelayMs, maxDelayMs, backoffMultiplier);

      // Log retry
      logger.warn(`${operationName} failed (attempt ${attempt + 1}/${maxAttempts}), retrying in ${delay}ms`, {
        error: error.message,
        attempt: attempt + 1,
        maxAttempts,
        delay,
      });

      // Call onRetry callback if provided
      if (onRetry) {
        await onRetry(error, attempt);
      }

      // Wait before retrying
      await sleep(delay);
    }
  }

  // This should never be reached, but TypeScript needs it
  throw lastError;
}

/**
 * Creates a retry wrapper for a function with preset options.
 *
 * @param {Object} defaultOptions - Default retry options
 * @returns {Function} Function that wraps another function with retry logic
 *
 * @example
 * const retryLLM = createRetryWrapper({
 *   maxAttempts: 3,
 *   operationName: 'LLM API call'
 * });
 *
 * const result = await retryLLM(async () => await callLLM());
 */
export function createRetryWrapper(defaultOptions = {}) {
  return async (fn, overrideOptions = {}) => {
    return withRetry(fn, { ...defaultOptions, ...overrideOptions });
  };
}

/**
 * Executes a function with a timeout.
 *
 * @template T
 * @param {Function} fn - Async function to execute
 * @param {number} timeoutMs - Timeout in milliseconds
 * @param {string} [operationName='operation'] - Name for error messages
 * @returns {Promise<T>} Result of the function
 * @throws {Error} Timeout error if function takes too long
 *
 * @example
 * const result = await withTimeout(
 *   async () => await slowOperation(),
 *   5000,
 *   'slowOperation'
 * );
 */
export async function withTimeout(fn, timeoutMs, operationName = 'operation') {
  return new Promise(async (resolve, reject) => {
    // Create timeout
    const timeoutId = setTimeout(() => {
      reject(new Error(`${operationName} timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    try {
      // Execute function
      const result = await fn();
      clearTimeout(timeoutId);
      resolve(result);
    } catch (error) {
      clearTimeout(timeoutId);
      reject(error);
    }
  });
}

/**
 * Combines retry logic with timeout.
 *
 * @template T
 * @param {Function} fn - Async function to execute
 * @param {Object} options - Combined options
 * @param {number} options.timeoutMs - Timeout in milliseconds
 * @param {Object} [options.retryOptions] - Retry options (see withRetry)
 * @param {string} [options.operationName] - Name of operation
 * @returns {Promise<T>} Result of the function
 *
 * @example
 * const result = await withRetryAndTimeout(
 *   async () => await apiCall(),
 *   {
 *     timeoutMs: 10000,
 *     retryOptions: { maxAttempts: 3 },
 *     operationName: 'API call'
 *   }
 * );
 */
export async function withRetryAndTimeout(fn, { timeoutMs, retryOptions = {}, operationName = 'operation' }) {
  return withRetry(
    async () => withTimeout(fn, timeoutMs, operationName),
    { ...retryOptions, operationName }
  );
}
