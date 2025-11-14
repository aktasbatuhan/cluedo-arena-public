/**
 * Human Player UI Controls
 * Handles user interaction for making suggestions, accusations, and challenge responses.
 */

let currentRequestType = null;
let currentRequestData = null;

// Initialize human player controls
function initializeHumanPlayerControls() {
  // Listen for requests from server
  socket.on('request-suggestion', handleSuggestionRequest);
  socket.on('request-accusation', handleAccusationRequest);
  socket.on('request-challenge-response', handleChallengeRequest);
  socket.on('player-eliminated', handlePlayerEliminated);

  // Create and inject control panel HTML
  const controlPanel = document.createElement('div');
  controlPanel.id = 'human-player-controls';
  controlPanel.className = 'hidden';
  controlPanel.innerHTML = `
    <div class="control-overlay">
      <div class="control-modal">
        <div class="control-header">
          <h2 id="control-title">Your Turn</h2>
          <div id="control-subtitle"></div>
        </div>

        <div class="control-content">
          <!-- Your Hand -->
          <div id="your-hand-section" class="section">
            <h3>Your Hand</h3>
            <div id="your-hand-cards" class="card-grid"></div>
          </div>

          <!-- Eliminated Cards -->
          <div id="eliminated-cards-section" class="section">
            <h3>Eliminated Cards</h3>
            <div id="eliminated-cards-list" class="card-grid"></div>
          </div>

          <!-- Suggestion Form -->
          <div id="suggestion-form" class="hidden">
            <div class="form-group">
              <label>Suspect</label>
              <select id="suspect-select" class="form-select"></select>
            </div>
            <div class="form-group">
              <label>Weapon</label>
              <select id="weapon-select" class="form-select"></select>
            </div>
            <div class="form-group">
              <label>Room (Current Location)</label>
              <input id="room-input" type="text" readonly class="form-input" />
            </div>
            <div class="form-group">
              <label>Reasoning (Optional)</label>
              <textarea id="reasoning-input" class="form-textarea" placeholder="Why are you making this suggestion?"></textarea>
            </div>
            <button id="submit-suggestion" class="btn-primary">Make Suggestion</button>
          </div>

          <!-- Accusation Form -->
          <div id="accusation-form" class="hidden">
            <p class="warning-text">⚠️ Warning: An incorrect accusation will eliminate you from the game!</p>
            <div class="accusation-options">
              <button id="accuse-yes" class="btn-danger">Make Accusation</button>
              <button id="accuse-no" class="btn-secondary">Don't Accuse (Continue Playing)</button>
            </div>
            <div id="accusation-details" class="hidden">
              <div class="form-group">
                <label>Suspect</label>
                <select id="accusation-suspect" class="form-select"></select>
              </div>
              <div class="form-group">
                <label>Weapon</label>
                <select id="accusation-weapon" class="form-select"></select>
              </div>
              <div class="form-group">
                <label>Room</label>
                <select id="accusation-room" class="form-select"></select>
              </div>
              <button id="submit-accusation" class="btn-danger">Confirm Accusation</button>
              <button id="cancel-accusation" class="btn-secondary">Cancel</button>
            </div>
          </div>

          <!-- Challenge Response Form -->
          <div id="challenge-form" class="hidden">
            <p>Choose which card to show:</p>
            <div id="matching-cards" class="card-select-grid"></div>
          </div>
        </div>
      </div>
    </div>
  `;

  document.body.appendChild(controlPanel);

  // Set up event listeners
  setupEventListeners();
}

function setupEventListeners() {
  // Suggestion form
  document.getElementById('submit-suggestion').addEventListener('click', submitSuggestion);

  // Accusation form
  document.getElementById('accuse-yes').addEventListener('click', () => {
    document.getElementById('accusation-details').classList.remove('hidden');
    document.querySelector('.accusation-options').classList.add('hidden');
  });

  document.getElementById('accuse-no').addEventListener('click', () => {
    socket.emit('human-accusation', {
      shouldAccuse: false,
      accusation: { suspect: null, weapon: null, room: null },
      reasoning: 'Chose not to accuse',
    });
    hideControls();
  });

  document.getElementById('submit-accusation').addEventListener('click', submitAccusation);
  document.getElementById('cancel-accusation').addEventListener('click', () => {
    document.getElementById('accusation-details').classList.add('hidden');
    document.querySelector('.accusation-options').classList.remove('hidden');
  });
}

// Handle suggestion request
function handleSuggestionRequest(data) {
  currentRequestType = 'suggestion';
  currentRequestData = data;

  // Show controls
  document.getElementById('human-player-controls').classList.remove('hidden');
  document.getElementById('control-title').textContent = 'Make Your Suggestion';
  document.getElementById('control-subtitle').textContent = `Turn ${data.gameState.currentTurn} - You are in the ${data.location}`;

  // Show your hand
  displayCards('your-hand-cards', data.cards, 'your-card');

  // Show eliminated cards
  displayCards('eliminated-cards-list', data.memory.eliminatedCards, 'eliminated-card');

  // Populate form
  populateSelect('suspect-select', data.gameState.availableSuspects);
  populateSelect('weapon-select', data.gameState.availableWeapons);
  document.getElementById('room-input').value = data.location;

  // Show suggestion form
  document.getElementById('suggestion-form').classList.remove('hidden');
  document.getElementById('accusation-form').classList.add('hidden');
  document.getElementById('challenge-form').classList.add('hidden');
}

// Handle accusation request
function handleAccusationRequest(data) {
  currentRequestType = 'accusation';
  currentRequestData = data;

  // Show controls
  document.getElementById('human-player-controls').classList.remove('hidden');
  document.getElementById('control-title').textContent = 'Make an Accusation?';
  document.getElementById('control-subtitle').textContent = `Turn ${data.gameState?.currentTurn || '?'}`;

  // Show your hand and eliminated cards
  displayCards('your-hand-cards', Array.from(data.memory.knownCards), 'your-card');
  displayCards('eliminated-cards-list', Array.from(data.memory.eliminatedCards), 'eliminated-card');

  // Reset accusation form
  document.getElementById('accusation-details').classList.add('hidden');
  document.querySelector('.accusation-options').classList.remove('hidden');

  // Show accusation form
  document.getElementById('suggestion-form').classList.add('hidden');
  document.getElementById('accusation-form').classList.remove('hidden');
  document.getElementById('challenge-form').classList.add('hidden');

  // Populate accusation selects (will be shown if user chooses to accuse)
  populateSelect('accusation-suspect', SUSPECTS);
  populateSelect('accusation-weapon', WEAPONS);
  populateSelect('accusation-room', ROOMS);
}

// Handle challenge request
function handleChallengeRequest(data) {
  currentRequestType = 'challenge';
  currentRequestData = data;

  // Show controls
  document.getElementById('human-player-controls').classList.remove('hidden');
  document.getElementById('control-title').textContent = 'Show a Card';
  document.getElementById('control-subtitle').textContent = `${data.suggestion.suspect}, ${data.suggestion.weapon}, ${data.suggestion.room}`;

  // Show matching cards as buttons
  const matchingCardsContainer = document.getElementById('matching-cards');
  matchingCardsContainer.innerHTML = '';

  data.matchingCards.forEach(card => {
    const cardButton = document.createElement('button');
    cardButton.className = 'card-button';
    cardButton.textContent = card;
    cardButton.onclick = () => submitChallengeResponse(card);
    matchingCardsContainer.appendChild(cardButton);
  });

  // Show challenge form
  document.getElementById('suggestion-form').classList.add('hidden');
  document.getElementById('accusation-form').classList.add('hidden');
  document.getElementById('challenge-form').classList.remove('hidden');
}

// Handle player elimination
function handlePlayerEliminated(data) {
  alert(data.message);
}

// Submit suggestion
function submitSuggestion() {
  const suggestion = {
    suspect: document.getElementById('suspect-select').value,
    weapon: document.getElementById('weapon-select').value,
    room: document.getElementById('room-input').value,
    reasoning: document.getElementById('reasoning-input').value || 'No reasoning provided',
  };

  if (!suggestion.suspect || !suggestion.weapon || !suggestion.room) {
    alert('Please fill in all fields');
    return;
  }

  socket.emit('human-suggestion', suggestion);
  hideControls();
}

// Submit accusation
function submitAccusation() {
  const accusation = {
    shouldAccuse: true,
    accusation: {
      suspect: document.getElementById('accusation-suspect').value,
      weapon: document.getElementById('accusation-weapon').value,
      room: document.getElementById('accusation-room').value,
    },
    reasoning: 'Human player accusation',
  };

  if (!accusation.accusation.suspect || !accusation.accusation.weapon || !accusation.accusation.room) {
    alert('Please fill in all fields');
    return;
  }

  if (!confirm('Are you sure? An incorrect accusation will eliminate you!')) {
    return;
  }

  socket.emit('human-accusation', accusation);
  hideControls();
}

// Submit challenge response
function submitChallengeResponse(card) {
  socket.emit('human-challenge-response', {
    cardToShow: card,
    reasoning: 'Human player choice',
  });
  hideControls();
}

// Helper functions
function displayCards(containerId, cards, className) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';

  if (!cards || cards.length === 0) {
    container.innerHTML = '<span class="no-cards">None</span>';
    return;
  }

  cards.forEach(card => {
    const cardEl = document.createElement('span');
    cardEl.className = `card ${className}`;
    cardEl.textContent = card;
    container.appendChild(cardEl);
  });
}

function populateSelect(selectId, options) {
  const select = document.getElementById(selectId);
  select.innerHTML = '';

  options.forEach(option => {
    const optionEl = document.createElement('option');
    optionEl.value = option;
    optionEl.textContent = option;
    select.appendChild(optionEl);
  });
}

function hideControls() {
  document.getElementById('human-player-controls').classList.add('hidden');
  currentRequestType = null;
  currentRequestData = null;
}

// Game constants (should match backend)
const SUSPECTS = [
  'Miss Scarlet',
  'Colonel Mustard',
  'Mrs. White',
  'Mr. Green',
  'Mrs. Peacock',
  'Professor Plum',
];

const WEAPONS = ['Candlestick', 'Knife', 'Lead Pipe', 'Revolver', 'Rope', 'Wrench'];

const ROOMS = [
  'Kitchen',
  'Ballroom',
  'Conservatory',
  'Dining Room',
  'Billiard Room',
  'Library',
  'Lounge',
  'Hall',
  'Study',
];

// Auto-initialize when loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeHumanPlayerControls);
} else {
  initializeHumanPlayerControls();
}
