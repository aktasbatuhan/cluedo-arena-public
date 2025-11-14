# Cluedo AI

---

![Predibase - The first reinforcement fine-tuning platform](assets/images/predibase.png)

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

## Deployment

Want to deploy Cluedo Arena to the cloud? Check out our comprehensive [**Deployment Guide**](DEPLOYMENT.md) which covers:

- 🚀 **Railway** (Recommended) - Best Socket.IO support with zero config
- 🎯 **Render** - Excellent alternative with free tier
- ⚠️ **Vercel** - Not recommended due to WebSocket limitations
- 📦 Environment variable setup
- 🔧 Troubleshooting tips

The deployment guide includes step-by-step instructions for each platform and explains why some platforms work better than others for this real-time application.

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

    ```text
    You are the Cluedo agent Red Agent. Your turn 3.
    Your hand: Miss Scarlet, Candlestick, Kitchen.
    Your current location: Lounge.
    Available Rooms: Kitchen, Ballroom, Conservatory, Dining Room, Billiard Room, Library, Lounge, Hall, Study

    Your knowledge:
    Known cards held: Miss Scarlet, Candlestick, Kitchen
    Eliminated Cards (Not in Solution): Colonel Mustard, Rope
    Suspected Cards: {}
    Current Deductions Summary: I know Colonel Mustard and Rope are not solution cards.
    Turn History Highlights:
    Blue Agent suggested: Mr. Green, Wrench, Library. Yellow Agent showed Blue Agent a card.

    Based on your knowledge and location (Lounge), make a strategic suggestion (suspect, weapon, room).
    The suggested room MUST be your current location: Lounge.
    Your goal is to gain new information.

    Respond ONLY with a YAML object in the following format. Provide concise reasoning.

    suspect: <string, one of available suspects>
    weapon: <string, one of available weapons>
    room: <string, MUST be Lounge>
    reasoning: <string, your detailed thought process>
    ```

*   **Example LLM YAML Response for Suggestion:**

    ```yaml
    suspect: Professor Plum
    weapon: Lead Pipe
    room: Lounge
    reasoning: "I am in the Lounge. Professor Plum and Lead Pipe are cards I don't have and haven't seen. Suggesting them might force another player to reveal if they hold one of these cards, or if no one challenges, it increases my suspicion for these cards."
    ```

**2. Updating Memory**

After each turn's events (suggestion and any challenge), all agents update their memory. The LLM is prompted to process these events and deduce new information.

*   **Example Memory Update Prompt (Simplified for an observing agent):**

    ```text
    You are Green Agent. Analyze the events from Red Agent's last turn (Turn 3) and update your memory and deductions.

    WHAT IS A DEDUCTION:
    A deduction is a card that you can definitively conclude is NOT part of the murder solution.

    Your current knowledge:
    Cards in my hand: Mr. Green, Library, Wrench
    Known Eliminated Cards: None
    Your most recent memory note: (No previous memory summary)

    Events from THIS turn:
    Red Agent suggested: Professor Plum, Lead Pipe, Lounge.
    Purple Agent showed Red Agent a card (you did not see the card).

    Based ONLY on the information above, what new cards can you definitively deduce are NOT part of the solution?
    Remember: A deduction must be 100% certain.

    Respond ONLY with a YAML object in the following format. Provide a DETAILED summary.

    newlyDeducedCards:
      - <string> # Card name, or empty list if none
    reasoning: <string> # Explain exactly how you deduced each new card
    memorySummary: <string> # DETAILED summary of your CURRENT understanding.
    ```

*   **Example LLM YAML Response for Memory Update:**

    ```yaml
    newlyDeducedCards: [] # As an observer, I didn't see the card Purple Agent showed.
    reasoning: "Red Agent made a suggestion. Purple Agent challenged it by showing a card to Red Agent. I don't know what card was shown, so I cannot deduce any new cards for certain this turn based on that specific challenge event alone."
    memorySummary: "My hand contains Mr. Green, Library, Wrench. Red Agent suggested Professor Plum, Lead Pipe, Lounge. Purple Agent holds at least one of these three cards. My suspicion for the combination of Professor Plum, Lead Pipe, Lounge as the solution decreases slightly, as at least one is held by Purple Agent."
    ```

This structured communication via YAML, combined with detailed prompting, allows the game engine to leverage the reasoning capabilities of different LLMs to play Cluedo. The `LoggingService` records all these interactions, which are then used by the data processing and evaluation scripts.

## Fine-Tuning for Deductive Reasoning

To enhance the deductive capabilities of LLMs within the Cluedo environment, this project supports fine-tuning using Predibase's Group Preference Optimization (GRPO).

**Process Overview:**

1.  **Data Preparation**:
    *   Game interaction logs, especially `memory_update` events and their corresponding `deduction_comparison` ground truths, are processed by the `scripts/data_processing/prepare_clue_memory_data.py` script.
    *   This script transforms raw logs into a structured JSONL format, where each line contains a detailed prompt (representing the agent's knowledge and current turn events) and the `ground_truth_deductions` (the cards the agent should logically deduce).
    *   The prompts are augmented with specific instructions and context relevant to the Cluedo memory update task.

2.  **Predibase GRPO Training**:
    *   The prepared JSONL dataset is uploaded to Predibase.
    *   The `scripts/training/predibase_clue_train.py` script is used to configure and launch a GRPO fine-tuning job on Predibase.
    *   **Reward Function**: A custom Python reward function (`calculate_memory_update_reward` within the script) is central to GRPO. This function:
        *   Parses the LLM's YAML completion to extract its `newlyDeducedCards`.
        *   Compares these predicted deductions against the `ground_truth_deductions` from the dataset.
        *   Calculates a reward based on the F1-score of this comparison, rewarding accurate and complete deductions while penalizing incorrect ones or badly formatted YAML.
    *   The GRPO process iteratively adjusts the base model (e.g., `qwen2-5-7b-instruct`) based on the rewards received, aiming to produce an adapter that excels at the Cluedo deduction task.
    *   The resulting fine-tuned adapter can then be deployed on Predibase and evaluated using scripts like `scripts/evaluation/evaluate_cluedo_model.py` or `scripts/evaluation/run_comprehensive_evaluations.py`.

This fine-tuning approach allows for specialized LLM agents that are better adapted to the specific reasoning patterns required in Cluedo. The sponsorship credits from **[Predibase](https://predibase.com/)** were instrumental in enabling this GRPO training.

## Extending with Custom LLMs

You can integrate additional LLM models by updating the `MODEL_LIST` in `cluedo_game_engine/src/services/llm.js` and ensuring the necessary API call logic and environment variables are in place.

## Requirements

- Node.js 14+
- NPM 6+
- Python 3.8+ (for running scripts)
- MongoDB instance (Optional, local or cloud-based like MongoDB Atlas)
- API Keys (refer to Configuration section):
- Cohere API key
    - OpenRouter API key (Optional, for access to a wider range of models)
    - Predibase API key (Optional, for using fine-tuned models on Predibase)


## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
## Running Tests

### Node.js Tests
Run the built-in Node test runner for JavaScript components:
```bash
cd cluedo_game_engine
npm test
```

### Python Tests
Run pytest for the Python utilities and scripts:
```bash
pytest
```
