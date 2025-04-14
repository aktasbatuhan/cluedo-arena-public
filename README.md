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

## Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/cluedo-arena.git
    cd cluedo-arena
    ```

2.  **Install Dependencies:**
    ```bash
    npm install
    ```

3.  **Configure Environment Variables:**
    Rename `.env.example` to `.env` and fill in the necessary values:
    ```bash
    # Edit .env to add your Cohere API key
    cp .env.example .env
    nano .env # Or your preferred editor
    ```

4.  **Start the Application:**
    ```bash
    npm start
    ```
    The application will be available at `http://localhost:3000` (or the port specified in your `.env` file).

## Configuration

All game configuration is done through environment variables:

- `COHERE_API_KEY`: Your API key for Cohere
- `SITE_URL`: Your site URL for logging or other services
- `SITE_NAME`: Your site name for logging or other services
- `MONGO_URI`: Connection string for your MongoDB database
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
- MongoDB instance (local or cloud-based like MongoDB Atlas)
- Cohere API key

## License

Open source - please add your preferred license

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Environment Variables

The application uses the following environment variables (defined in `.env`):

-   `COHERE_API_KEY`: Your API key for Cohere. Get one from [cohere.com](https://cohere.com/).
-   `SITE_URL`: (Optional) Your site URL, might be used for logging or other services.
-   `SITE_NAME`: (Optional) Your site name, might be used for logging or other services.
-   `MONGO_URI`: Connection string for your MongoDB database.
-   `PORT`: Port number for the server to run on (default: 3000).
-   `MAX_TURNS`: Maximum number of turns allowed in a game (default: 120).
-   `LOG_LEVEL`: Controls log verbosity. Options are 'error', 'warn', 'info' (default), or 'debug' (most verbose).
    - Use `LOG_LEVEL=debug` when running to see detailed LLM responses and parsing information.
    - Example: `LOG_LEVEL=debug npm run run-games`
-   **LLM Integration:** Utilizes Large Language Models (currently via Cohere API) for agent decision-making.
-   **Game Logic:** Implements the core rules and mechanics of Cluedo.
-   **Agent Memory:** Basic implementation for agents to remember cards and deductions.
-   **MongoDB:** Stores game results and potentially agent performance data.
-   **Logging:** Detailed logging of game events and LLM interactions.

## Future Enhancements
