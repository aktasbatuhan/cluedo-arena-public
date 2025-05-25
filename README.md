# Cluedo AI

---

![Predibase - The first reinforcement fine-tuning platform](predibase.png)

Sponsored by **[Predibase](https://predibase.com/)** - The
Enterprise Low-Code AI Platform.
Predibase empowers developers and data scientists to build, fine-tune, and deploy custom AI models with ease.
This work was made possible with credits granted by Predibase for GRPO training.

---

A platform for evaluating LLM reasoning capabilities through the classic deduction game Cluedo (Clue).

## The Challenge of AI Social Reasoning

Human collaboration hinges on sophisticated social reasoning and Theory of Mind (ToM) — the ability to infer and track what others know, believe, and intend. For Large Language Models (LLMs) to become truly effective partners in complex, multi-agent environments, they must develop analogous capabilities. This includes inferring hidden mental states from observed actions, asking pertinent questions to gather missing information, and dynamically maintaining and updating their understanding of the world and other agents within it.

Despite significant strides in areas like instruction-following, current LLMs often exhibit limitations in robust multi-agent reasoning and strategic interaction. This project, **Cluedo AI**, aims to address a core question: *Can current LLMs demonstrate human-like social reasoning and memory utilization within a structured, turn-based multi-agent setting with a finite, yet complex, deduction space?*

### Why Clue as a Benchmark?

The classic game of Cluedo (Clue) was chosen as the testbed for this research due to its unique properties that mirror aspects of real-world social deduction:

-   **Interactive & Dynamic:** Each turn generates new information, both explicitly stated and subtly implied.
-   **Memory-Intensive:** Success depends critically on remembering past events, suggestions, and challenges.
-   **Multi-Agent Dynamics:** Requires players (LLMs) to reason about the knowledge, intentions, and potential deceptions of other players. Actions are often contingent on beliefs about others' beliefs.
-   **Evaluable at Each Step:** The game's structure allows for the assessment of an LLM's deductions and decisions at every turn against a ground truth.
-   **Grounded in Human-like Social Deduction:** The core mechanics resonate with how humans approach problems involving incomplete information and social inference.

## Overview

Cluedo AI provides a robust platform to empirically investigate these questions. It establishes a controlled environment where multiple LLMs compete as agents in Cluedo. By observing and analyzing how different models perform in this game of memory, deduction, and social strategy, we can systematically evaluate their reasoning capabilities.

## Features

- **Multi-LLM Competition**: Multiple LLMs (Claude, GPT-4o, Gemini, Llama, etc.) compete against each other
- **Memory System**: Each agent maintains their own memory of game events and deductions
- **Strategic Decision Making**: Agents decide when to make suggestions, how to respond to challenges, and when to risk making accusations
- **Metrics Collection**: Track win rates, game completion times, risk aversion scores, and more
- **Web Visualization**: Optional web interface for watching games in progress
- **Leaderboard**: Compare performance across LLM models

## Project Structure

Here's a walkthrough of the key directories and files in this project:

*   **`cluedo_game_engine/`**: Contains the core Node.js application for the Cluedo game.
    *   **`public/`**: Static assets for the web visualization (HTML, CSS, client-side JavaScript).
        *   `index.html`: The main page for the web interface.
        *   `app.js`: Client-side logic for handling game events and updating the UI.
        *   `style.css`: Styles for the web interface.
    *   **`src/`**: Source code for the game engine.
        *   `config/gameConstants.js`: Defines core game items like suspects, weapons, and rooms.
        *   `models/`
            *   `Agent.js`: Defines the AI agent class, including its memory and decision-making calls to the LLM service.
            *   `Game.js`: Manages the game state, rules, turn progression, and agent interactions.
            *   `GameResult.js`: Handles the structure and saving of game results.
            *   `Memory.js`: Represents an agent's memory system.
        *   `services/`
            *   `llm.js`: The core service for interacting with various Large Language Models (Cohere, OpenRouter, Predibase). Handles prompt construction, API calls, response parsing, and fallback logic.
            *   `LoggingService.js`: Manages logging of LLM interactions and game events.
        *   `utils/logger.js`: Configuration for the Winston logging library.
        *   `gameLogic.js`: Contains the main loop for running a game.
        *   `server.js`: Sets up the Express web server, Socket.IO for real-time communication, and handles game mode selection (web UI vs. batch).
        *   `index.js`: Entry point for the Node.js application.
    *   `package.json`: Defines Node.js project metadata, dependencies, and scripts.
    *   `.env` (not committed, created from `.env.example`): Stores environment variables like API keys and database URIs.

*   **`scripts/`**: Contains Python scripts for various tasks.
    *   **`data_processing/`**: Scripts to prepare and transform data.
        *   `prepare_clue_memory_data.py`: Generates training data for Cluedo memory deduction.
        *   `prepare_dria_batch.py`: Creates batch files for Dria inference.
        *   `convert_batch_files.py`: Converts Dria batch files to a new format.
        *   `sample_jsonl.py`: Utility to sample lines from JSONL files.
    *   **`evaluation/`**: Scripts for evaluating LLM performance.
        *   `evaluate_cluedo_model.py`: Evaluates a fine-tuned Predibase model on Cluedo tasks.
        *   `run_comprehensive_evaluations.py`: Runs Cluedo evaluations across multiple LLM providers.
        *   `process_and_visualize_evaluations.py`: Processes raw evaluation CSVs, calculates metrics, and generates charts.
        *   Other scripts for specific provider evaluations (Cohere, OpenRouter, Dria).
    *   **`training/`**: Scripts related to model training.
        *   `predibase_clue_train.py`: Fine-tunes a model on Predibase using GRPO for Cluedo memory deduction.
    *   `plot_model_summary.py`: Utility to plot metrics from a summary CSV.
    *   **`utility/`**: Other helper scripts.

*   **`README.md`**: This file, providing an overview of the project.
*   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore.
*   **`requirements.txt`**: Lists Python dependencies for the scripts.

## Running Cluedo Games

Cluedo AI offers several ways to run and observe games, catering to different needs from interactive visualization to large-scale batch evaluations.

### Interactive UI Mode
To watch games unfold in real-time, you can use the web-based UI. Start the application using:
```bash
npm start
```
The game interface will be accessible at `http://localhost:3000` (or the port specified in your `.env` file). The UI displays the game board, agent actions, suggestions, challenges, and key game log events. Static assets for the UI are located in `cluedo_game_engine/public/`, with client-side logic in `app.js`.

### Batch Mode for Evaluation & Data Collection
For systematic evaluations or generating training data, games can be run in batch mode without the UI. This is essential for running many games efficiently. Python scripts in the `scripts/evaluation/` directory, such as `run_comprehensive_evaluations.py`, are used for this purpose. These scripts typically handle game setup, iteration, and results aggregation.

Consult the individual scripts for specific arguments and configurations for batch runs.
Example:
```bash
# Ensure your .env file is configured with necessary API keys
python scripts/evaluation/run_comprehensive_evaluations.py --num_games 100 --output_file my_comprehensive_eval.csv
```

### LLM Backend Configuration
The game engine is designed to work with various LLM providers and models. The core LLM interaction logic resides in `cluedo_game_engine/src/services/llm.js`.

-   Supported providers include Cohere, OpenRouter (which provides access to a wide range of models like GPT, Claude, Llama, etc.), and fine-tuned models deployed on Predibase.
-   API keys for these services (e.g., `COHERE_API_KEY`, `OPENROUTER_API_KEY`, `PREDIBASE_API_KEY`) must be configured in your `.env` file (see Configuration section).
-   The specific models used by agents can often be configured within the evaluation scripts or potentially through environment variables, depending on the execution mode.

### Dria Batch Inference for Scaled Evaluations
For very large-scale evaluations, this project supports preparing batch input files for Dria's Batch Inference service ([`https://dria.co/batch-inference`](https://dria.co/batch-inference)). This allows for massive parallel processing of game scenarios or deduction tasks.

-   The script `scripts/data_processing/prepare_dria_batch.py` can be used to prepare data for Dria. Output files would be generated in a user-specified location.

## Quick Start

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/aktasbatuhan/cluedo-arena-public
    cd cluedo-arena
    ```

2.  **Install Dependencies:**
    *   **Node.js Backend:**
        ```bash
        cd cluedo_game_engine
        npm install
        cd ..
        ```
    *   **Python Scripts:**
        ```bash
        pip install -r requirements.txt
        ```

3.  **Configure Environment Variables:**
    Navigate to the `cluedo_game_engine` directory and rename `.env.example` to `.env`. Fill in the necessary API keys and other configurations.
    ```bash
    cd cluedo_game_engine
    cp .env.example .env
    nano .env # Or your preferred editor (e.g., vim, code)
    cd ..
    ```
    See the "Configuration" section below for details on the required variables.

4.  **Start the Application (UI Mode):**
    ```bash
    cd cluedo_game_engine
    npm start
    ```
    The application will be available at `http://localhost:3000` (or the port specified in your `.env` file).

## Configuration

Key configurations are managed through environment variables in the `cluedo_game_engine/.env` file:

-   `PORT`: Port number for the server to run on (default: 3000).
-   `MAX_TURNS`: Maximum number of turns allowed in a game (default: 120).
-   `MONGO_URI`: (Optional) Connection string for your MongoDB database if you wish to store game results and logs persistently.
-   `LOG_LEVEL`: Controls log verbosity for the Node.js application. Options: 'error', 'warn', 'info' (default), 'debug'.
    -   Example: `LOG_LEVEL=debug`
-   **API Keys (Essential for LLM interaction):**
    -   `COHERE_API_KEY`: Your API key for Cohere.
    -   `OPENROUTER_API_KEY`: Your API key for OpenRouter, providing access to various models.
    -   `PREDIBASE_API_KEY`: Your API key for Predibase (if using fine-tuned models deployed on Predibase).
    -   *(Add other API keys as needed for new integrations)*
-   `SITE_URL`, `SITE_NAME`: (Optional) Might be used by specific logging configurations or services.

Ensure these are set before running the application or evaluation scripts.

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

## LLM Interaction: A Closer Look

The core of Cluedo AI involves intricate interactions with Large Language Models (LLMs) to simulate agent decision-making and memory. Here's a glimpse into how these interactions occur, using YAML for structured communication:

**1. Making a Suggestion**

When it's an agent's turn to make a suggestion, the game engine constructs a detailed prompt providing the LLM with the necessary context.

*   **Example Suggestion Prompt (Simplified):**

    ```