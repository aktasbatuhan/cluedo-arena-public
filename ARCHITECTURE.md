# Cluedo Arena - Architecture Documentation

## Overview

Cluedo Arena is a platform for evaluating LLM reasoning capabilities through the classic deduction game Cluedo (Clue). The system enables multiple LLM agents to compete against each other in a structured environment.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface                          │
│              (Socket.IO + Express Server)                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│                      Game Engine                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Game.js     │  │  Agent.js    │  │  Memory.js   │     │
│  │              │  │              │  │              │     │
│  │ - State mgmt │  │ - Decisions  │  │ - Tracking   │     │
│  │ - Turn logic │  │ - Strategy   │  │ - Deduction  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
└────────────────────────────┼────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                      LLM Service Layer                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              LLM Service (llm.js)                    │  │
│  │  - Request routing                                   │  │
│  │  - Prompt engineering                                │  │
│  │  - Response parsing                                  │  │
│  │  - Retry logic                                       │  │
│  └──────┬───────────────────┬───────────────────────────┘  │
│         │                   │                              │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────────────┐  │
│  │  Cohere     │    │ OpenRouter  │    │  Predibase   │  │
│  │  Provider   │    │  Provider   │    │  Provider    │  │
│  └─────────────┘    └─────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│                     External APIs                           │
│  - Cohere API                                               │
│  - OpenRouter API (GPT, Claude, Gemini, etc.)               │
│  - Predibase API (Fine-tuned models)                        │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Game Engine (`cluedo_game_engine/src/`)

#### 1. Game.js
**Purpose**: Orchestrates the entire game flow

**Responsibilities**:
- Game initialization (solution creation, card distribution)
- Turn management and progression
- Agent coordination
- Event logging and broadcasting
- Victory condition checking

**Key Methods**:
- `createSolution()`: Creates the murder solution
- `distributeCards()`: Distributes cards among agents
- `processTurn()`: Handles complete turn sequence
- `processChallenges()`: Manages challenge phase
- `checkAccusation()`: Validates accusations
- `updateAgentMemories()`: Coordinates memory updates

**State Management**:
```javascript
{
  solution: { suspect, weapon, room },
  agents: Array<Agent>,
  currentTurn: number,
  activeAgentIndex: number,
  gameLog: Array<Event>,
  turnHistory: Array<TurnData>,
  isOver: boolean,
  winner: Agent | null
}
```

#### 2. Agent.js
**Purpose**: Represents an AI agent playing the game

**Responsibilities**:
- Decision making (suggestions, accusations, challenges)
- LLM interaction coordination
- Memory management
- Location tracking

**Key Methods**:
- `makeSuggestion(gameState)`: Makes strategic suggestion
- `considerAccusation(suggestion, challengeResult)`: Decides on accusation
- `respondToChallenge(suggestion)`: Responds to challenges
- `move(availableRooms)`: Moves to new room

**State**:
```javascript
{
  name: string,
  cards: Set<string>,
  model: string,
  hasLost: boolean,
  memory: Memory,
  location: string
}
```

#### 3. Memory.js
**Purpose**: Manages agent's knowledge and deductions

**Responsibilities**:
- Track known, suspected, and eliminated cards
- Maintain turn history
- Format memory for LLM consumption
- Update based on game events

**Key Data Structures**:
```javascript
{
  knownCards: Set<string>,           // Cards in hand
  eliminatedCards: Set<string>,      // Proven not in solution
  suspectedCards: Map<string, number>, // Card -> confidence
  playerNegativeConstraints: Map<string, Set<string>>, // Player -> cards they don't have
  currentMemory: string,             // LLM-generated summary
  memoryHistory: Array<TurnEvent>    // Historical events
}
```

**Key Methods**:
- `formatMemoryForLLM()`: Formats state for LLM prompts
- `recordTurnEvents()`: Records turn events
- `update(summary, deducedCards)`: Updates with LLM insights
- `maintain()`: Memory cleanup and optimization

### Service Layer

#### 4. LLM Service (`services/llm.js`)
**Purpose**: Central hub for all LLM interactions

**Responsibilities**:
- Route requests to appropriate providers
- Construct prompts for different tasks
- Parse and validate LLM responses
- Handle retries and timeouts
- Fallback logic

**Task Types**:
1. **Suggestion**: Generate suspect/weapon/room suggestion
2. **Memory Update**: Process turn events and update memory
3. **Challenge**: Decide which card to show
4. **Accusation**: Decide whether to make final accusation

**Prompt Engineering**:
Each task type has specialized prompt construction:
- Context about current game state
- Agent's memory and knowledge
- Specific instructions and constraints
- YAML response format requirements

**Response Handling**:
```javascript
{
  // LLM response parsed from YAML
  parsedResponse: Object,

  // Request ID for logging
  requestId: string,

  // Error information if applicable
  error: Error | null
}
```

#### 5. Logging Service (`services/LoggingService.js`)
**Purpose**: Tracks all LLM interactions for analysis

**Responsibilities**:
- Log LLM requests and responses
- Record deduction comparisons
- Save to JSON files for evaluation

### Utility Modules

#### Configuration (`config/appConfig.js`)
Centralized configuration with validation:
- Server settings (port, environment)
- Game settings (max turns, agent count)
- LLM settings (backend, timeout, retry config)
- API keys management

#### Error Handling (`utils/errors.js`)
Custom error classes for different failure modes:
- `AppError`: Base application error
- `LLMError`: LLM API failures
- `LLMTimeoutError`: Request timeouts
- `LLMValidationError`: Invalid LLM responses
- `GameValidationError`: Invalid game state
- `NetworkError`: Network failures

#### Retry Logic (`utils/retry.js`)
Robust retry mechanism with:
- Exponential backoff
- Jitter to prevent thundering herd
- Configurable retry policies
- Timeout handling
- Retryability detection

#### Validation (`utils/validation.js`)
Input validation for:
- Card names
- Suggestions
- Accusations
- Memory updates
- Challenge responses

## Data Flow

### Turn Execution Flow

```
1. Game.processTurn()
   │
   ├─> Agent.move() - Move to room
   │
   ├─> Agent.makeSuggestion()
   │   ├─> Memory.formatMemoryForLLM()
   │   ├─> LLMService.makeSuggestion()
   │   │   ├─> Construct prompt
   │   │   ├─> Call LLM API (with retry)
   │   │   └─> Parse YAML response
   │   └─> Return suggestion
   │
   ├─> Game.processChallenges()
   │   └─> For each agent:
   │       ├─> Agent.respondToChallenge()
   │       ├─> LLMService.challenge()
   │       └─> Show card if possible
   │
   ├─> Game.updateAgentMemories()
   │   └─> For each agent:
   │       ├─> Memory.recordTurnEvents()
   │       ├─> LLMService.updateMemory()
   │       │   ├─> Construct memory prompt
   │       │   ├─> Call LLM API
   │       │   └─> Parse deductions
   │       └─> Memory.update()
   │
   ├─> Agent.considerAccusation()
   │   ├─> LLMService.considerAccusation()
   │   └─> Return accusation decision
   │
   └─> If accusation:
       └─> Game.checkAccusation()
           └─> Determine winner or eliminate agent
```

### Memory Update Flow

```
Turn Event Occurs
   │
   ├─> Memory.recordTurnEvents() - Store raw events
   │
   ├─> LLMService.updateMemory()
   │   │
   │   ├─> Build prompt with:
   │   │   ├─> Current memory state
   │   │   ├─> Known cards
   │   │   ├─> Eliminated cards
   │   │   ├─> Turn events (who suggested what, who challenged)
   │   │
   │   ├─> LLM reasons about events
   │   │   └─> Returns YAML:
   │   │       ├─> newlyDeducedCards: []
   │   │       ├─> reasoning: "..."
   │   │       └─> memorySummary: "..."
   │   │
   │   └─> Parse and validate response
   │
   └─> Memory.update()
       ├─> Append summary to currentMemory
       ├─> Add deduced cards to eliminatedCards
       └─> Update timestamp
```

## Configuration

### Environment Variables

See `.env.example` for complete list. Key variables:

```bash
# Server
PORT=3000
MAX_TURNS=120

# LLM
LLM_BACKEND=OPENROUTER
LLM_REQUEST_TIMEOUT=60000

# API Keys
COHERE_API_KEY=...
OPENROUTER_API_KEY=...
PREDIBASE_API_KEY=...

# Logging
LOG_LEVEL=info
```

### Model List

Models configured in `llm.js`:
```javascript
MODEL_LIST = [
  'anthropic/claude-3.5-sonnet',
  'cohere/command-a',
  'cohere/command-r-plus',
  'google/gemini-2.5-flash-preview',
  'openai/gpt-4o-2024-11-20'
]
```

## Game Constants

Defined in `config/gameConstants.js`:

```javascript
SUSPECTS = [
  'Miss Scarlet', 'Colonel Mustard', 'Mrs. White',
  'Mr. Green', 'Mrs. Peacock', 'Professor Plum'
]

WEAPONS = [
  'Candlestick', 'Knife', 'Lead Pipe',
  'Revolver', 'Rope', 'Wrench'
]

ROOMS = [
  'Kitchen', 'Ballroom', 'Conservatory',
  'Dining Room', 'Billiard Room', 'Library',
  'Lounge', 'Hall', 'Study'
]
```

## Testing

### Test Structure

Tests located in `/tests/`:
- `Agent.test.js` - Agent functionality
- `Memory.test.js` - Memory management
- `Game.test.js` - Game mechanics
- `errors.test.js` - Error handling
- `retry.test.js` - Retry logic

### Running Tests

```bash
cd cluedo_game_engine
npm test              # Run all tests
npm run test:watch    # Watch mode
```

## Deployment Modes

### 1. Interactive UI Mode
```bash
npm start
```
- Web interface at http://localhost:3000
- Real-time game visualization
- Socket.IO events for live updates

### 2. Batch Mode
```bash
node src/index.js --run-games --num-games=10
```
- No UI
- Runs multiple games sequentially
- Results saved to files

### 3. Evaluation Mode
```bash
python scripts/evaluation/run_comprehensive_evaluations.py
```
- Large-scale testing
- Metrics collection
- Performance comparison

## Performance Considerations

### Memory Management
- Memory history limited to 100 entries
- Memory summary capped at 10KB
- Automatic truncation of old entries

### LLM Request Optimization
- Timeout: 60 seconds default
- Retry: 3 attempts with exponential backoff
- Concurrent request limiting (TBD)

### Logging
- Structured JSON logging
- Daily log rotation
- Configurable log levels

## Security Considerations

1. **API Keys**: Never commit to git, use environment variables
2. **Input Validation**: All LLM responses validated before use
3. **Rate Limiting**: Consider implementing for production
4. **Timeout Enforcement**: Prevent hanging requests

## Future Improvements

1. **LLM Service Refactoring**: Split into provider-specific modules
2. **Enhanced Testing**: Integration and E2E tests
3. **Performance Monitoring**: Metrics collection and dashboards
4. **Caching**: Cache LLM responses for deterministic testing
5. **Database Integration**: MongoDB for persistent storage
6. **WebSocket Optimization**: Reduce event payload sizes
7. **TypeScript Migration**: Incremental migration for type safety

## References

- Game Rules: See README.md
- API Documentation: (TBD)
- Contributing Guidelines: (TBD)
