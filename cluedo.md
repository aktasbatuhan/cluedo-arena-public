# AI Cluedo: LLM-Powered Mystery Game
## Product Requirements Document
### Version 2.0

## Overview
AI Cluedo is a web-based implementation of the classic board game Cluedo where AI agents powered by Large Language Models (LLMs) play against each other. The game focuses on natural language interactions between agents and their ability to maintain and update memory of game events.

## Core Game Mechanics

### Game Setup
1. Game Components
   - 6 AI agents (characters)
   - 6 weapons
   - 9 rooms
   - One solution envelope (containing murderer, weapon, location)
   - Remaining cards distributed among agents

2. AI Agent Initialization
   - Each agent receives:
     - Set of cards
     - Initial memory state (empty)
     - Turn order assignment

### Turn Structure

#### Active Agent Turn
1. Suggestion Phase
   - Active agent proposes a murder combination:
     - Suspect
     - Weapon
     - Room
   - System formats suggestion as natural language
   - Suggestion broadcasted to all agents

2. Challenge Phase
   - Other agents check their cards against suggestion
   - Agents with contradicting cards must challenge
   - First agent (clockwise) with contradicting evidence shows one card
   - Challenge response formatted in natural language

3. Memory Update Phase
   - All agents receive:
     - Active agent's suggestion
     - Challenge results (if any)
     - Turn metadata (turn number, active agent, etc.)

### Agent Memory System

#### Memory Structure
```javascript
{
  agentId: String,
  currentMemory: String,  // Free-format text
  turnNumber: Number,
  lastUpdated: Timestamp
}
```

#### Memory Update Process
1. Input Collection
   - Previous memory state
   - New turn information:
     - Active player's suggestion
     - Challenge results
     - Any shown cards
     - Other agent reactions

2. Memory Generation
   - Agent processes all inputs using LLM
   - Generates new free-format text memory
   - Memory should include:
     - Important deductions
     - Known card locations
     - Suspected information
     - Strategic considerations

3. Memory Storage
   - New memory replaces old memory
   - Previous memory used as context for future updates
   - System maintains memory history for analysis

### Technical Requirements

#### Agent Interaction System
1. Turn Management
   ```javascript
   {
     turnNumber: Number,
     activeAgent: String,
     suggestion: {
       suspect: String,
       weapon: String,
       room: String,
       narrativeText: String
     },
     challenges: [{
       challengingAgent: String,
       shownCard: String,
       narrativeText: String
     }]
   }
   ```

2. LLM Integration
   - Prompt structure for suggestions
   - Prompt structure for challenges
   - Prompt structure for memory updates
   - Context management system

#### Memory Management System
1. Memory Update Pipeline
   ```javascript
   {
     previousMemory: String,
     newInformation: {
       turnData: Object,
       observedActions: Array,
       deductions: Array
     },
     updatedMemory: String
   }
   ```

2. Storage Requirements
   - Memory versioning
   - Turn-by-turn history
   - Retrieval system
   - Backup system

### User Interface Requirements

#### Game State Display
1. Current Turn Information
   - Active agent
   - Suggestion details
   - Challenge results
   - Turn number

2. Agent Memory Viewer
   - Current memory state
   - Memory history
   - Memory update visualization
   - Analysis tools

#### Interaction Logging
1. Turn Log
   - Chronological event listing
   - Natural language descriptions
   - Important deductions
   - Challenge results

2. Analysis Tools
   - Memory evolution tracking
   - Decision analysis
   - Strategy visualization
   - Agent comparison


## Implementation Requirements

### LLM Integration
1. Prompt Engineering
   - Suggestion generation
   - Challenge evaluation
   - Memory updates
   - Natural language responses

2. Context Management
   - Token optimization
   - Memory summarization
   - Important information retention
   - Context window management

### Testing Requirements
1. Functional Testing
   - Turn mechanics
   - Memory updates
   - Challenge system
   - Game rules

2. Performance Testing
   - Response times
   - Memory efficiency
   - System stability
   - Error handling

## Sign-off Requirements
- Game logic validation
- Memory system testing
- Performance benchmarks
- User interface testing
- Documentation review