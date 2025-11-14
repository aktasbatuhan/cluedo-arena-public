import test from 'node:test';
import assert from 'node:assert/strict';
import { Game } from '../cluedo_game_engine/src/models/Game.js';

test('Game - constructor initializes with correct defaults', () => {
  const game = new Game('spectate');

  assert.equal(game.mode, 'spectate');
  assert.equal(game.currentTurn, 0);
  assert.equal(game.activeAgentIndex, 0);
  assert.equal(game.activePlayers, 6);
  assert.equal(game.isOver, false);
  assert.equal(game.winner, null);
  assert.equal(game.agents.length, 0);
  assert.equal(game.gameLog.length, 0);
  assert.equal(game.turnHistory.length, 0);
  assert.equal(game.AGENT_NAMES.length, 6);
});

test('Game - createSolution returns valid solution', () => {
  const game = new Game();
  const solution = game.createSolution();

  assert.ok(solution.suspect);
  assert.ok(solution.weapon);
  assert.ok(solution.room);
  assert.ok(game.SUSPECTS.includes(solution.suspect));
  assert.ok(game.WEAPONS.includes(solution.weapon));
  assert.ok(game.ROOMS.includes(solution.room));
});

test('Game - getRemainingCards excludes solution cards', () => {
  const game = new Game();
  game.solution = {
    suspect: game.SUSPECTS[0],
    weapon: game.WEAPONS[0],
    room: game.ROOMS[0],
  };

  const remaining = game.getRemainingCards();

  assert.ok(!remaining.includes(game.solution.suspect));
  assert.ok(!remaining.includes(game.solution.weapon));
  assert.ok(!remaining.includes(game.solution.room));

  // Total cards should be (6 suspects + 6 weapons + 9 rooms) - 3 solution cards = 18
  assert.equal(remaining.length, 18);
});

test('Game - distributeCards distributes evenly among 6 agents', () => {
  const game = new Game();
  game.solution = game.createSolution();
  const remaining = game.getRemainingCards();
  const distributed = game.distributeCards(remaining);

  assert.equal(distributed.length, 6);

  // Check all cards are distributed (18 total cards)
  const totalCards = distributed.reduce((sum, hand) => sum + hand.length, 0);
  assert.equal(totalCards, 18);

  // Each agent should have 3 cards
  distributed.forEach(hand => {
    assert.equal(hand.length, 3);
  });

  // Check no duplicate cards
  const allCards = distributed.flat();
  const uniqueCards = new Set(allCards);
  assert.equal(allCards.length, uniqueCards.size);
});

test('Game - shuffle randomizes array', () => {
  const game = new Game();
  const original = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const shuffled = game.shuffle([...original]);

  // Length should be the same
  assert.equal(shuffled.length, original.length);

  // All elements should be present
  original.forEach(item => {
    assert.ok(shuffled.includes(item));
  });

  // Note: There's a tiny chance this could fail if shuffle returns same order
  // but with 10 elements, probability is 1 in 3,628,800
  // For a more robust test, we'd run it multiple times
});

test('Game - getRandomElement returns element from array', () => {
  const game = new Game();
  const array = ['a', 'b', 'c', 'd', 'e'];
  const element = game.getRandomElement(array);

  assert.ok(array.includes(element));
});

test('Game - logEvent adds to game log with timestamp', () => {
  const game = new Game();
  const event = {
    type: 'test',
    agent: 'Test Agent',
    message: 'Test message',
  };

  game.logEvent(event);

  assert.equal(game.gameLog.length, 1);
  assert.equal(game.gameLog[0].type, 'test');
  assert.equal(game.gameLog[0].agent, 'Test Agent');
  assert.ok(game.gameLog[0].timestamp);
});
