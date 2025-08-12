import test from 'node:test';
import assert from 'node:assert/strict';
import { Memory } from '../cluedo_game_engine/src/models/Memory.js';

test('addKnownCard updates sets correctly', () => {
  const memory = new Memory('agent1');
  memory.suspectedCards.set('Candlestick', 0.5);
  memory.eliminatedCards.add('Candlestick');
  memory.addKnownCard('Candlestick');
  assert.ok(memory.knownCards.has('Candlestick'));
  assert.ok(!memory.suspectedCards.has('Candlestick'));
  assert.ok(!memory.eliminatedCards.has('Candlestick'));
});

test('doesPlayerNotHaveCard reflects negative constraints', () => {
  const memory = new Memory('agent2');
  memory.playerNegativeConstraints.set('Alice', new Set(['Rope']));
  assert.ok(memory.doesPlayerNotHaveCard('Alice', 'Rope'));
  assert.ok(!memory.doesPlayerNotHaveCard('Alice', 'Knife'));
});

test('reset clears memory state', () => {
  const memory = new Memory('agent3');
  memory.addKnownCard('Lead Pipe');
  memory.suspectedCards.set('Wrench', 0.7);
  memory.eliminatedCards.add('Revolver');
  memory.playerNegativeConstraints.set('Bob', new Set(['Library']));
  memory.currentMemory = 'some memory';
  memory.memoryHistory.push({ turnNumber: 1 });
  memory.reset();
  assert.equal(memory.knownCards.size, 1);
  assert.ok(memory.knownCards.has('Lead Pipe'));
  assert.equal(memory.suspectedCards.size, 0);
  assert.equal(memory.eliminatedCards.size, 0);
  assert.equal(memory.playerNegativeConstraints.size, 0);
  assert.equal(memory.currentMemory, '');
  assert.equal(memory.memoryHistory.length, 0);
});
