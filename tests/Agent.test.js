import test from 'node:test';
import assert from 'node:assert/strict';
import { Agent } from '../cluedo_game_engine/src/models/Agent.js';

test('constructor stores initial cards in memory', () => {
  const agent = new Agent('Alice', ['Knife', 'Rope'], 'gpt', 'game1');
  assert.ok(agent.cards instanceof Set);
  assert.ok(agent.memory.knownCards.has('Knife'));
  assert.ok(agent.memory.knownCards.has('Rope'));
});

test('setLost marks agent as lost and updates memory', () => {
  const agent = new Agent('Bob', [], 'gpt', 'game1');
  agent.setLost();
  assert.ok(agent.hasLost);
  assert.ok(agent.memory.currentMemory.includes('incorrect accusation'));
});
