# Cluedo AI

A platform for evaluating LLM reasoning capabilities through the classic deduction game Cluedo (Clue).

## Overview

Cluedo AI creates a controlled environment where multiple language learning models (LLMs) compete in the classic board game Cluedo (known as Clue in North America). By comparing how different models perform in a game that requires memory, deduction, and strategic thinking, we can evaluate their reasoning capabilities in a structured way.

## Features

- **Multi-LLM Competition**: Multiple LLMs (Claude, GPT-4o, Gemini, Llama, etc.) compete against each other
- **Memory System**: Each agent maintains their own memory of game events and deductions
- **Strategic Decision Making**: Agents decide when to make suggestions, how to respond to challenges, and when to risk making accusations
- **Metrics Collection**: Track win rates, game completion times, risk aversion scores, and more
- **Web Visualization**: Optional web interface for watching games in progress
- **Leaderboard**: Compare performance across LLM models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cluedo-ai.git
cd cluedo-ai

# Install dependencies
npm install

# Set up your environment variables
cp .env.example .env
# Edit .env to add your OpenRouter API key
```

## Usage

```bash
# Run a single game
npm start

# Run multiple games for better statistical comparison
node src/index.js --games 10
```

## Configuration

All game configuration is done through environment variables:

- `OPENROUTER_API_KEY`: Your API key for OpenRouter
- `SITE_URL`: Your site URL for OpenRouter
- `SITE_NAME`: Your site name for OpenRouter
- `PORT`: Server port for web visualization (default: 3000)
- `MAX_TURNS`: Maximum turns per game (default: 120)

## How It Works

1. **Game Setup**: 
   - Random solution is selected (suspect, weapon, room)
   - Remaining cards are distributed among agents
   - Each agent is assigned an LLM model

2. **Turn Structure**:
   - Agent makes a suggestion (suspect, weapon, room)
   - Other agents may challenge by showing a card that contradicts the suggestion
   - Agent may make an accusation based on gathered information
   - All agents update their memory with information from the turn

3. **Memory System**:
   - Each agent maintains its own memory of game events
   - Memory includes known cards, eliminated cards, and deductions
   - Agents use LLMs to interpret game events and update their memory

4. **Victory Conditions**:
   - Correct accusation: Agent wins
   - Incorrect accusation: Agent is eliminated
   - Last agent standing: Wins by default
   - Max turns reached: Game ends with no winner

## Extending

You can add your own LLM models by modifying the `MODEL_LIST` array in `src/services/llm.js`.

## Requirements

- Node.js 14+
- NPM 6+
- OpenRouter API key

## License

Open source - please add your preferred license

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.