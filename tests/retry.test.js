import test from 'node:test';
import assert from 'node:assert/strict';
import { withRetry, withTimeout, withRetryAndTimeout } from '../cluedo_game_engine/src/utils/retry.js';

test('withRetry - succeeds on first attempt', async () => {
  let attempts = 0;
  const result = await withRetry(async () => {
    attempts++;
    return 'success';
  }, { maxAttempts: 3 });

  assert.equal(result, 'success');
  assert.equal(attempts, 1);
});

test('withRetry - succeeds after retries', async () => {
  let attempts = 0;
  const result = await withRetry(async () => {
    attempts++;
    if (attempts < 3) {
      const error = new Error('Temporary failure');
      error.code = 'ECONNRESET'; // Retryable error
      throw error;
    }
    return 'success';
  }, {
    maxAttempts: 5,
    initialDelayMs: 10,
    maxDelayMs: 100,
  });

  assert.equal(result, 'success');
  assert.equal(attempts, 3);
});

test('withRetry - fails after max attempts', async () => {
  let attempts = 0;
  await assert.rejects(
    async () => {
      await withRetry(async () => {
        attempts++;
        const error = new Error('Persistent failure');
        error.code = 'ETIMEDOUT'; // Retryable error
        throw error;
      }, {
        maxAttempts: 3,
        initialDelayMs: 10,
      });
    },
    {
      message: 'Persistent failure',
    }
  );

  assert.equal(attempts, 3);
});

test('withRetry - does not retry non-retryable errors', async () => {
  let attempts = 0;
  await assert.rejects(
    async () => {
      await withRetry(async () => {
        attempts++;
        throw new Error('Non-retryable error');
      }, {
        maxAttempts: 3,
        shouldRetry: () => false,
      });
    },
    {
      message: 'Non-retryable error',
    }
  );

  assert.equal(attempts, 1);
});

test('withTimeout - succeeds within timeout', async () => {
  const result = await withTimeout(
    async () => {
      await new Promise(resolve => setTimeout(resolve, 10));
      return 'success';
    },
    100,
    'test operation'
  );

  assert.equal(result, 'success');
});

test('withTimeout - fails when timeout exceeded', async () => {
  await assert.rejects(
    async () => {
      await withTimeout(
        async () => {
          await new Promise(resolve => setTimeout(resolve, 200));
          return 'success';
        },
        50,
        'slow operation'
      );
    },
    {
      message: 'slow operation timed out after 50ms',
    }
  );
});

test('withRetryAndTimeout - combines retry and timeout', async () => {
  let attempts = 0;
  const result = await withRetryAndTimeout(
    async () => {
      attempts++;
      if (attempts < 2) {
        const error = new Error('Retry me');
        error.code = 'ECONNRESET';
        throw error;
      }
      return 'success';
    },
    {
      timeoutMs: 1000,
      retryOptions: {
        maxAttempts: 3,
        initialDelayMs: 10,
      },
      operationName: 'combined test',
    }
  );

  assert.equal(result, 'success');
  assert.equal(attempts, 2);
});
