import test from 'node:test';
import assert from 'node:assert/strict';
import { LLMService, MODEL_LIST } from '../cluedo_game_engine/src/services/llm.js';
import { providerFactory } from '../cluedo_game_engine/src/services/llm/ProviderFactory.js';
import { PromptBuilder } from '../cluedo_game_engine/src/services/llm/prompts/PromptBuilder.js';
import { responseParser } from '../cluedo_game_engine/src/services/llm/parsing/ResponseParser.js';

test('MODEL_LIST is exported and contains models', () => {
  assert.ok(Array.isArray(MODEL_LIST));
  assert.ok(MODEL_LIST.length > 0);
  assert.ok(MODEL_LIST.includes('anthropic/claude-3.5-sonnet'));
});

test('LLMService has required static methods', () => {
  assert.ok(typeof LLMService.initialize === 'function');
  assert.ok(typeof LLMService.makeSuggestion === 'function');
  assert.ok(typeof LLMService.updateMemory === 'function');
  assert.ok(typeof LLMService.evaluateChallenge === 'function');
  assert.ok(typeof LLMService.considerAccusation === 'function');
});

test('ProviderFactory can list providers', () => {
  const providers = providerFactory.listProviders();
  assert.ok(Array.isArray(providers));
  // May be empty if no API keys are configured
});

test('ProviderFactory has required methods', () => {
  assert.ok(typeof providerFactory.getProvider === 'function');
  assert.ok(typeof providerFactory.getProviderForModel === 'function');
  assert.ok(typeof providerFactory.getDefaultProvider === 'function');
  assert.ok(typeof providerFactory.hasProvider === 'function');
});

test('PromptBuilder has all prompt building methods', () => {
  assert.ok(typeof PromptBuilder.buildSuggestionPrompt === 'function');
  assert.ok(typeof PromptBuilder.buildMemoryUpdatePrompt === 'function');
  assert.ok(typeof PromptBuilder.buildChallengePrompt === 'function');
  assert.ok(typeof PromptBuilder.buildAccusationPrompt === 'function');
});

test('ResponseParser can parse and validate responses', () => {
  assert.ok(typeof responseParser.parse === 'function');
  assert.ok(typeof responseParser.parseSuggestion === 'function');
  assert.ok(typeof responseParser.parseAccusation === 'function');
  assert.ok(typeof responseParser.parseMemoryUpdate === 'function');
  assert.ok(typeof responseParser.parseChallenge === 'function');
});

test('ResponseParser validates suggestion correctly', () => {
  const validYaml = `
suspect: Miss Scarlet
weapon: Candlestick
room: Kitchen
reasoning: Strategic choice
`;

  const result = responseParser.parse(validYaml, 'suggestion');
  assert.ok(result.valid);
  assert.equal(result.data.suspect, 'Miss Scarlet');
  assert.equal(result.data.weapon, 'Candlestick');
  assert.equal(result.data.room, 'Kitchen');
});

test('ResponseParser rejects invalid suggestion', () => {
  const invalidYaml = `
suspect: Invalid Person
weapon: Candlestick
room: Kitchen
`;

  const result = responseParser.parse(invalidYaml, 'suggestion');
  assert.ok(!result.valid);
  assert.ok(result.error);
});

test('ResponseParser validates memory update correctly', () => {
  const validYaml = `
newlyDeducedCards:
  - Knife
  - Library
reasoning: These cards were shown
memorySummary: I now know these are eliminated
`;

  const result = responseParser.parse(validYaml, 'memoryUpdate');
  assert.ok(result.valid);
  assert.ok(Array.isArray(result.data.newlyDeducedCards));
  assert.equal(result.data.newlyDeducedCards.length, 2);
});

test('ResponseParser handles markdown-wrapped YAML', () => {
  const yamlWithMarkdown = `\`\`\`yaml
suspect: Colonel Mustard
weapon: Rope
room: Lounge
reasoning: Test
\`\`\``;

  const result = responseParser.parse(yamlWithMarkdown, 'suggestion');
  assert.ok(result.valid);
  assert.equal(result.data.suspect, 'Colonel Mustard');
});
