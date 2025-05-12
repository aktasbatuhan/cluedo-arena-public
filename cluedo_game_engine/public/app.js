const socket = io();

// Game constants
const SUSPECTS = [
  'Miss Scarlet',
  'Colonel Mustard',
  'Mrs. White',
  'Mr. Green',
  'Mrs. Peacock',
  'Professor Plum'
];

const WEAPONS = [
  'Candlestick',
  'Dagger',
  'Lead Pipe',
  'Revolver',
  'Rope',
  'Wrench'
];

const ROOMS = [
  'Kitchen',
  'Ballroom',
  'Conservatory',
  'Dining Room',
  'Billiard Room',
  'Library',
  'Lounge',
  'Hall',
  'Study'
];

// Game state
let state = {
  activeAgent: '',
  agents: [],
  recentSuggestions: []
};

socket.on('game-log', (log) => {
  const logContainer = document.getElementById('log-messages');
  log.forEach(entry => addLogEntry(entry));
});

socket.on('game-state', (state) => {
  updateGameState(state);
});

socket.on('game-event', (event) => {
  addLogEntry(event);
  updatePlayers(event.agents || [], state.activeAgent);

  switch (event.type) {
    case 'GAME_OVER':
      addToGameLog(`${formatTimestamp(event.timestamp)} ${event.winner.name} won the game!`);
      endGame();
      break;
    default:
      // Handle other event types
      break;
  }
});

let isHumanPlayer = false;
let playerIndex = -1;

function showSuggestionForm() {
  const form = document.getElementById('suggestion-form');
  form.innerHTML = `
    <div class="form-content">
      <h4>Make a Suggestion</h4>
      <select id="suspect-select">
        <option value="">Select Suspect</option>
        ${SUSPECTS.map(s => `<option value="${s}">${s}</option>`).join('')}
      </select>
      <select id="weapon-select">
        <option value="">Select Weapon</option>
        ${WEAPONS.map(w => `<option value="${w}">${w}</option>`).join('')}
      </select>
      <select id="room-select">
        <option value="">Select Room</option>
        ${ROOMS.map(r => `<option value="${r}">${r}</option>`).join('')}
      </select>
      <div class="form-buttons">
        <button onclick="submitSuggestion()">Submit</button>
        <button onclick="hideForm('suggestion-form')">Cancel</button>
      </div>
    </div>
  `;
  form.classList.remove('hidden');
}

function hideForm(formId) {
  document.getElementById(formId).classList.add('hidden');
}

function submitSuggestion() {
  const suspect = document.getElementById('suspect-select').value;
  const weapon = document.getElementById('weapon-select').value;
  const room = document.getElementById('room-select').value;

  if (!suspect || !weapon || !room) {
    alert('Please select all fields');
    return;
  }

  socket.emit('player-action', {
    type: 'suggestion',
    suspect,
    weapon,
    room
  });

  hideForm('suggestion-form');
}

socket.on('assign-player', (index) => {
  isHumanPlayer = true;
  playerIndex = index;
  document.getElementById('player-controls').classList.remove('hidden');
});

function addLogEntry(entry) {
  const logContainer = document.getElementById('log-messages');
  const entryDiv = document.createElement('div');
  entryDiv.className = 'log-entry';
  entryDiv.setAttribute('data-type', entry.type);
  
  const timestamp = new Date(entry.timestamp).toLocaleTimeString();
  const content = `
    <div class="log-header">
      <span class="timestamp">${timestamp}</span>
      <strong class="agent">${entry.agent || 'System'}</strong>
      <span class="event-type">${entry.type}</span>
    </div>
    <div class="log-content">${entry.narrativeText || entry.message || JSON.stringify(entry.details)}</div>
  `;
  
  entryDiv.innerHTML = content;
  logContainer.appendChild(entryDiv);
  logContainer.scrollTop = logContainer.scrollHeight;
}

function updatePlayers(agents, activeAgent) {
  const playersList = document.getElementById('players-list');
  playersList.innerHTML = agents.map(agent => `
    <div class="player ${agent.hasLost ? 'eliminated' : ''}" 
         data-current="${agent.name === activeAgent}">
      <span class="agent-name">${agent.name}</span>
      <span class="agent-status">${agent.hasLost ? '❌ Eliminated' : '🕵️ Active'}</span>
      ${isHumanPlayer ? '' : `
      <div class="card-tooltip">
        <h3>${agent.name}'s Cards</h3>
        <ul>
          ${agent.cards?.map(card => `<li>${card}</li>`).join('') || '<li>No cards shown</li>'}
        </ul>
      </div>`}
    </div>
  `).join('');
}

function updateGameState(state) {
  updatePlayers(state.agents, state.activeAgent);
  updateCurrentTurn(state.activeAgent);
  updateTurnOrder(state.agents, state.activeAgentIndex);
  updatePlayerPositions(state.agents);
  updateRoomHighlights(state.recentSuggestions);
  
  // Update player controls visibility
  if (isHumanPlayer) {
    const playerControls = document.getElementById('player-controls');
    if (state.activeAgent === 'Human Player') {
      playerControls.classList.remove('hidden');
    } else {
      playerControls.classList.add('hidden');
    }
  }
  
  // Check for winner
  if (state.winner) {
    showWinner(state);
  }
}

function updatePlayerPositions(agents) {
  document.querySelectorAll('.room').forEach(room => {
    room.querySelectorAll('.player-marker').forEach(marker => marker.remove());
    
    const roomName = room.dataset.room;
    agents.filter(agent => !agent.hasLost).forEach(agent => {
      if (agent.position === roomName) {
        const marker = document.createElement('div');
        marker.className = `player-marker ${agent.name.toLowerCase().replace(' ', '-')}`;
        marker.textContent = agent.name[0];
        marker.title = `${agent.name} is here`;
        room.appendChild(marker);
      }
    });
  });
}

function updateRoomHighlights(suggestions) {
  document.querySelectorAll('.room').forEach(room => {
    const roomName = room.dataset.room.toLowerCase();
    room.classList.toggle('recent-suggestion', 
      suggestions?.some(s => s.toLowerCase() === roomName)
    );
  });
}

function updateCurrentTurn(activeAgent) {
  document.querySelector('#current-turn span').textContent = activeAgent;
}

function updateTurnOrder(agents, activeIndex) {
  const orderList = document.getElementById('turn-order-list');
  const nextPlayers = [];
  let current = activeIndex;
  let count = 0;
  let attempts = 0;
  
  while (count < 5 && attempts < 12) { // Max 2 full rotations
    current = (current + 1) % 6;
    attempts++;
    
    const agent = agents[current];
    if (!agent.hasLost && current !== activeIndex) {
      nextPlayers.push(agent.name);
      count++;
    }
  }

  orderList.innerHTML = nextPlayers
    .map(name => `<div class="turn-order-agent">${name}</div>`)
    .join('');
}

function showWinner(state) {
  const overlay = document.getElementById('winner-overlay');
  const winnerAgent = overlay.querySelector('.winner-agent');
  const winnerSolution = overlay.querySelector('.winner-solution');
  const gameStats = overlay.querySelector('.game-stats');
  
  winnerAgent.innerHTML = `🎉 ${state.winner} wins!`;
  
  winnerSolution.innerHTML = `
    <p>The correct solution was:</p>
    <p><strong>Suspect:</strong> ${state.solution.suspect}</p>
    <p><strong>Weapon:</strong> ${state.solution.weapon}</p>
    <p><strong>Room:</strong> ${state.solution.room}</p>
  `;
  
  gameStats.innerHTML = `
    <p>Total Turns: ${state.currentTurn}</p>
    <p>Active Players: ${state.activePlayers}</p>
    <p>Game Duration: ${formatGameDuration(state.startTime)}</p>
  `;
  
  overlay.classList.remove('hidden');
}

function formatGameDuration(startTime) {
  const duration = Math.floor((Date.now() - new Date(startTime)) / 1000);
  const minutes = Math.floor(duration / 60);
  const seconds = duration % 60;
  return `${minutes}m ${seconds}s`;
}

function startGame(mode) {
  // Initialize game state
  state = {
    activeAgent: '',
    agents: [],
    recentSuggestions: []
  };

  // Hide mode selection
  document.getElementById('mode-selection').classList.add('hidden');
  
  // Small delay to ensure smooth transition
  setTimeout(() => {
    // Show game container
    document.getElementById('game-container').classList.remove('hidden');
    
    // Notify server of mode choice
    socket.emit('game-mode', mode);
    
    // If playing, show controls
    if (mode === 'play') {
      document.getElementById('player-controls').classList.remove('hidden');
    }
  }, 300);
}

socket.on('private-challenge', (data) => {
  if (data.forAgent === 'Human Player') {
    const cardHistory = document.getElementById('shown-card-history');
    const entry = document.createElement('div');
    entry.className = 'shown-card-entry';
    
    // Handle cases where no card is shown
    const cardText = data.card || 'Could not determine card'; 
    
    entry.innerHTML = `
      <span class="challenger">${data.challengingAgent}</span>
      <span class="card">${cardText}</span>
      <span class="time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;
    
    cardHistory.prepend(entry);
    
    // Auto-scroll to top
    cardHistory.scrollTop = 0;
  }
});

socket.on('game-over', (data) => {
  console.log('Game Over:', data);
  const gameContainer = document.getElementById('game-container');
  const winnerMessage = document.createElement('div');
  winnerMessage.className = 'winner-message';
  winnerMessage.innerHTML = `
    <h2>Game Over!</h2>
    <p>${data.winner ? `Winner: ${data.winner}` : 'No winner'}</p>
    <p>Solution: ${data.solution.suspect} in the ${data.solution.room} with the ${data.solution.weapon}</p>
  `;
  gameContainer.appendChild(winnerMessage);
}); 