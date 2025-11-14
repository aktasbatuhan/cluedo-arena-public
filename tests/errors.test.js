import test from 'node:test';
import assert from 'node:assert/strict';
import {
  AppError,
  LLMError,
  LLMTimeoutError,
  LLMValidationError,
  ConfigurationError,
  GameValidationError,
  NetworkError,
  isRetryableError,
  fromAxiosError,
} from '../cluedo_game_engine/src/utils/errors.js';

test('AppError - creates error with default values', () => {
  const error = new AppError('Test error');

  assert.equal(error.message, 'Test error');
  assert.equal(error.name, 'AppError');
  assert.equal(error.statusCode, 500);
  assert.equal(error.isOperational, true);
  assert.deepEqual(error.context, {});
  assert.ok(error.timestamp);
});

test('AppError - creates error with custom values', () => {
  const error = new AppError('Custom error', {
    statusCode: 404,
    isOperational: false,
    context: { id: 123 },
  });

  assert.equal(error.statusCode, 404);
  assert.equal(error.isOperational, false);
  assert.deepEqual(error.context, { id: 123 });
});

test('AppError - toJSON returns correct structure', () => {
  const error = new AppError('Test error', {
    statusCode: 400,
    context: { foo: 'bar' },
  });

  const json = error.toJSON();

  assert.equal(json.name, 'AppError');
  assert.equal(json.message, 'Test error');
  assert.equal(json.statusCode, 400);
  assert.equal(json.isOperational, true);
  assert.deepEqual(json.context, { foo: 'bar' });
  assert.ok(json.timestamp);
  assert.ok(json.stack);
});

test('LLMError - includes LLM-specific context', () => {
  const error = new LLMError('LLM failed', {
    provider: 'openai',
    model: 'gpt-4',
    taskType: 'suggestion',
    attempt: 2,
  });

  assert.equal(error.name, 'LLMError');
  assert.equal(error.context.provider, 'openai');
  assert.equal(error.context.model, 'gpt-4');
  assert.equal(error.context.taskType, 'suggestion');
  assert.equal(error.context.attempt, 2);
});

test('LLMTimeoutError - has correct status code', () => {
  const error = new LLMTimeoutError('Request timed out');
  assert.equal(error.statusCode, 504);
  assert.equal(error.name, 'LLMTimeoutError');
});

test('LLMValidationError - includes validation details', () => {
  const error = new LLMValidationError('Invalid response', {
    response: { foo: 'bar' },
    schema: { type: 'object' },
    validationErrors: ['missing field'],
  });

  assert.equal(error.statusCode, 422);
  assert.deepEqual(error.context.response, { foo: 'bar' });
  assert.deepEqual(error.context.schema, { type: 'object' });
  assert.deepEqual(error.context.validationErrors, ['missing field']);
});

test('ConfigurationError - is non-operational', () => {
  const error = new ConfigurationError('Bad config');
  assert.equal(error.isOperational, false);
});

test('GameValidationError - includes game context', () => {
  const error = new GameValidationError('Invalid move', {
    agentName: 'Red Agent',
    turnNumber: 5,
  });

  assert.equal(error.statusCode, 400);
  assert.equal(error.context.agentName, 'Red Agent');
  assert.equal(error.context.turnNumber, 5);
});

test('NetworkError - includes network context', () => {
  const originalError = new Error('Connection failed');
  const error = new NetworkError('Network issue', {
    url: 'https://api.example.com',
    method: 'POST',
    originalError,
  });

  assert.equal(error.statusCode, 503);
  assert.equal(error.context.url, 'https://api.example.com');
  assert.equal(error.context.method, 'POST');
  assert.equal(error.context.originalError, 'Connection failed');
});

test('isRetryableError - NetworkError is retryable', () => {
  const error = new NetworkError('Network failed');
  assert.ok(isRetryableError(error));
});

test('isRetryableError - LLMTimeoutError is retryable', () => {
  const error = new LLMTimeoutError('Timed out');
  assert.ok(isRetryableError(error));
});

test('isRetryableError - LLMError with 429 is retryable', () => {
  const error = new LLMError('Rate limited', { statusCode: 429 });
  assert.ok(isRetryableError(error));
});

test('isRetryableError - LLMValidationError is not retryable', () => {
  const error = new LLMValidationError('Invalid');
  assert.ok(!isRetryableError(error));
});

test('isRetryableError - error with retryable code is retryable', () => {
  const error = new Error('Connection reset');
  error.code = 'ECONNRESET';
  assert.ok(isRetryableError(error));
});

test('fromAxiosError - handles timeout', () => {
  const axiosError = new Error('Timeout');
  axiosError.code = 'ECONNABORTED';

  const error = fromAxiosError(axiosError, { provider: 'test' });

  assert.ok(error instanceof LLMTimeoutError);
  assert.equal(error.context.provider, 'test');
});

test('fromAxiosError - handles response error', () => {
  const axiosError = new Error('Server error');
  axiosError.response = {
    status: 500,
    data: { error: { message: 'Internal error' } },
  };

  const error = fromAxiosError(axiosError);

  assert.ok(error instanceof LLMError);
  assert.equal(error.message, 'Internal error');
  assert.equal(error.statusCode, 500);
});

test('fromAxiosError - handles no response', () => {
  const axiosError = new Error('No response');
  axiosError.request = {};

  const error = fromAxiosError(axiosError);

  assert.ok(error instanceof NetworkError);
  assert.equal(error.message, 'No response received from server');
});
