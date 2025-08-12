const socket = io();

// Game state
let state = {
  activeAgent: '',
  agents: [],
  recentSuggestions: []
};
const pastGames = [];

// --- Socket Event Listeners ---

socket.on('game-log', (data) => {
    pastGames.push(data);
    renderPastGames();
});

socket.on('past-games', (games) => {
    pastGames.push(...games);
    renderPastGames();
});

socket.on('leaderboard-update', (stats) => {
  const tbody = document.querySelector('#leaderboard tbody');
  if (!tbody) return;
  const rows = Object.values(stats)
    .sort((a, b) => b.win_rate - a.win_rate) // Assuming win_rate exists
    .map(entry => `
      <tr>
        <td>${entry.model_name || entry.Model}</td>
        <td>${entry.games_played || entry.games}</td>
        <td>${entry.games_won || entry.wins}</td>
        <td>${(entry.win_rate * 100).toFixed(1) || (entry.games_won / entry.games_played * 100).toFixed(1)}%</td>
        <td>${entry.avg_completion_time?.toFixed(1) || 'N/A'}</td>
      </tr>
    `)
    .join('');
  tbody.innerHTML = rows;
});

socket.on('game-state', (newState) => {
  updateGameState(newState);
});

socket.on('game-event', (event) => {
  addLogEntry(event);
});

socket.on('memory-update', (data) => {
  const panel = document.getElementById('memory-panel');
  if (!panel) return;
  let section = panel.querySelector(`[data-agent="${data.agent}"]`);
  if (!section) {
    section = document.createElement('div');
    section.className = 'agent-memory';
    section.setAttribute('data-agent', data.agent);
    section.innerHTML = `<h3>${data.agent}</h3><pre class="memory-summary"></pre>`;
    panel.appendChild(section);
  }
  section.querySelector('.memory-summary').textContent = JSON.stringify(data.summary, null, 2);
});

socket.on('game-progress', ({ total, completed }) => {
  const overlay = document.getElementById('loading-overlay');
  if (!overlay.classList.contains('active')) return;

  const percent = (completed / total) * 100;
  const progress = document.querySelector('#progress-bar .progress');
  const status = document.getElementById('progress-status');

  progress.style.width = `${percent}%`;
  status.textContent = `Completed ${completed} of ${total} games...`;

  if (completed >= total) {
    // Hide overlay after a short delay
    setTimeout(() => {
        overlay.classList.remove('active');
    }, 500);
  }
});

// --- Game Functions ---

function startGame(mode) {
  document.getElementById('mode-selection').classList.add('hidden');
  document.getElementById('game-container').classList.remove('hidden');

  const data = { mode };

  if (mode === 'multi') {
    document.querySelectorAll('.multi-game-only').forEach(el => el.classList.remove('hidden'));

    // Show loading overlay
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.add('active');
    document.querySelector('#progress-bar .progress').style.width = '0%';
    document.getElementById('progress-status').textContent = 'Starting games...';

    // Get game count
    const input = document.getElementById('multi-count');
    const count = parseInt(input?.value, 10);
    if (!isNaN(count) && count > 0) {
      data.count = count;
    }
  }

  socket.emit('game-mode', data);
}

function addLogEntry(entry) {
  const logContainer = document.getElementById('log-messages');
  const entryDiv = document.createElement('div');
  entryDiv.className = 'log-entry';
  
  const timestamp = new Date(entry.timestamp).toLocaleTimeString();
  let content = `[${timestamp}] `;

  if (entry.type === 'SUGGESTION') {
    content += `${entry.agent} suggests: ${entry.suggestion.suspect}, ${entry.suggestion.weapon}, ${entry.suggestion.room}.`;
    if(entry.reasoning) content += ` <em>Reasoning: ${entry.reasoning}</em>`;
  } else if (entry.type === 'CHALLENGE') {
    content += `${entry.agent} showed a card to disprove the suggestion.`;
  } else if (entry.type === 'ACCUSATION') {
    content += `${entry.agent} accuses! Result: ${entry.message}`;
  } else {
    content += `${entry.agent || 'System'}: ${entry.message}`;
  }
  entryDiv.innerHTML = content;
  logContainer.prepend(entryDiv);
}

function updateGameState(newState) {
  state = newState;
  const playersList = document.getElementById('players-list');
  playersList.innerHTML = state.agents.map(agent => `
    <div class="player ${!agent.isAlive ? 'eliminated' : ''} ${agent.name === state.activeAgent ? 'current-turn' : ''}">
      <div class="player-header">
        <span class="agent-name">${agent.name} (${agent.model})</span>
        <span class="status-icon">${agent.isAlive ? 'Active' : 'Eliminated'}</span>
      </div>
      <div class="agent-cards">
        ${agent.cards ? agent.cards.map(card => `<span class="card">${card}</span>`).join(' ') : ''}
      </div>
    </div>
  `).join('');

  document.querySelector('#current-turn .current-number').textContent = state.currentTurn;
  if(state.solution) {
      document.getElementById('solution-display').textContent = `${state.solution.suspect} in the ${state.solution.room} with the ${state.solution.weapon}`;
  }
}

function renderPastGames() {
  const list = document.getElementById('past-games-list');
  list.innerHTML = pastGames.map((game, idx) => {
    const winnerInfo = game.winner ? `${game.winner.name} (${game.winner.model})` : 'No winner';
    const lines = game.log.map(entry => {
      const time = new Date(entry.timestamp).toLocaleTimeString();
      let line = `[${time}] ${entry.agent || 'System'} - ${entry.type}`;
      if (entry.message) line += `: ${entry.message}`;
      if (entry.reasoning) line += `\n  Reasoning: ${entry.reasoning}`;
      if (entry.summary) line += `\n  Memory: ${JSON.stringify(entry.summary, null, 2)}`;
      return line;
    }).join('\n');
    return `<details><summary>Game ${idx + 1} - Winner: ${winnerInfo}</summary><pre class="game-log">${lines}</pre></details>`;
  }).join('');
}