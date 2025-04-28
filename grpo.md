# Running GRPO Training with ART Server on Lightning AI Studio

## Overview

This guide explains how to run the GRPO (Game Reward Policy Optimization) training script located within the `cluedo-arena-public` project. This training process requires a separate local LLM server, referred to as the "ART Server" (presumably located in the `art-llm-server` directory), to be running concurrently. The GRPO script will communicate with the ART server to get LLM responses during the training simulation.

We will use two separate terminals within the Lightning AI Studio environment: one for the ART server and one for the GRPO training script.

## Prerequisites

1.  **Lightning AI Studio:** You need an active Lightning AI Studio instance.
2.  **Project Code:** Ensure you have both the `cluedo-arena-public` and `art-llm-server` projects cloned or uploaded into your Lightning AI Studio workspace filesystem.
3.  **Dependencies:** Make sure the necessary dependencies for both projects are installed. This usually involves:
    *   For `cluedo-arena-public` (likely Node.js): Running `npm install` in its root directory.
    *   For `art-llm-server` (likely Python): Running `pip install -r requirements.txt` in its root directory.
4.  **Environment Variables:** Check if either project requires specific environment variables set in a `.env` file (e.g., API keys, although less likely for a local ART server). The most critical ones for this process are `LLM_BACKEND` and `ART_WRAPPER_URL` for the `cluedo-arena-public` project.
5.  **Model:** Ensure the `clue-tiny-grpo` model files required by the `art-llm-server` are present within its directory structure as expected.

## Step-by-Step Instructions

**1. Open Two Terminals**

*   In your Lightning AI Studio interface, open two separate terminal tabs. We'll refer to them as **Terminal 1 (ART Server)** and **Terminal 2 (GRPO Training)**.

**2. Setup and Start the ART LLM Server (Terminal 1)**

*   **Navigate:** In **Terminal 1**, navigate to the directory containing the ART server code.
    ```bash
    cd path/to/your/art-llm-server
    ```
*   **Install Dependencies (if not already done):**
    ```bash
    # Assuming Python and requirements.txt
    pip install -r requirements.txt
    ```
*   **Run the Server:** Start the ART server script. The exact command might vary based on the project's setup (e.g., it could be `app.py`, `server.py`, `main.py`). Check the `art-llm-server`'s README if unsure.
    ```bash
    # Example: Replace 'app.py' with the actual server script name
    python app.py
    ```
*   **Verify:** The server should start and output messages indicating it's running and listening for connections, typically on a specific port (e.g., `localhost:5001`). **Leave this terminal running.** It needs to stay active throughout the GRPO training process.

**3. Configure and Start GRPO Training (Terminal 2)**

*   **Navigate:** In **Terminal 2**, navigate to the root directory of the main Cluedo project.
    ```bash
    cd path/to/your/cluedo-arena-public
    ```
*   **Install Dependencies (if not already done):**
    ```bash
    # Assuming Node.js and package.json
    npm install
    ```
*   **Configure Environment:** The GRPO training script needs to know it should use the ART backend and where to find it. Set the following environment variables *before* running the script:
    ```bash
    export LLM_BACKEND=ART
    export ART_WRAPPER_URL=http://localhost:5001
    ```
    *   **Note:** Ensure `ART_WRAPPER_URL` matches the host and port the ART server is listening on (usually `http://localhost:5001` if running within the same Studio environment). You might also configure these in a `.env` file in the `cluedo-arena-public` directory if the application loads it automatically.
*   **Run the Training Script:** Execute the GRPO training script. The command will depend on how the project is structured (e.g., a script in `src/training/`, or an `npm` script).
    ```bash
    # Example: Replace with the actual command to start GRPO training
    # Option A: Direct node execution
    node src/training/run_grpo.js # Adjust path/filename as needed

    # Option B: Using npm script (check package.json)
    # npm run train:grpo
    ```
*   **Verify Connection:** Watch the initial output in **Terminal 2**. It should indicate it's connecting to the ART backend. You should also see corresponding log messages (requests being received) in **Terminal 1** (ART Server) shortly after the training starts interacting with the LLM.

**4. Monitor Training**

*   **Terminal 1 (ART Server):** Will show logs for incoming requests from the GRPO script and the responses it sends back. Monitor for any errors here.
*   **Terminal 2 (GRPO Training):** Will display the main training loop output, such as game progress, turn numbers, agent actions, rewards, loss values, and potentially evaluation metrics.

**5. Stopping the Process**

*   **Stop GRPO:** When you want to stop the training, press `Ctrl+C` in **Terminal 2**.
*   **Stop ART Server:** Once the training script has finished or been stopped, you can stop the ART server by pressing `Ctrl+C` in **Terminal 1**.

## Key Configuration Points

*   **`LLM_BACKEND=ART`:** This environment variable tells the `cluedo-arena-public` application (specifically the `LLMService`) to route requests through the ART wrapper logic instead of directly to Cohere, OpenRouter, etc.
*   **`ART_WRAPPER_URL=http://localhost:5001`:** This tells the `LLMService` the exact URL where the ART server is listening for requests. Ensure the port (`5001`) matches the one used by your `art-llm-server` script. If the ART server runs on a different port, update this URL accordingly.

## Troubleshooting

*   **Connection Refused Errors (GRPO Terminal):**
    *   Verify the ART server is running in Terminal 1.
    *   Double-check that the `ART_WRAPPER_URL` in Terminal 2 exactly matches the host and port shown in Terminal 1's startup messages.
    *   Ensure no firewall rules within the Studio environment are blocking connections on the specified port (usually not an issue for `localhost` connections).
*   **Errors in ART Server Terminal:** Check the specific error message. It could be related to model loading, request formatting, or resource issues.
*   **Dependency Errors:** Ensure `npm install` and `pip install -r requirements.txt` completed successfully in their respective project directories.
*   **Incorrect Script Names/Paths:** Double-check the actual filenames for the ART server (`app.py`?) and the GRPO training script (`src/training/run_grpo.js`?) and adjust the commands accordingly.

## Conclusion

By running the ART server in one terminal and the correctly configured GRPO training script in another, you can perform training runs that leverage the local `clue-tiny-grpo` model served via the ART interface within your Lightning AI Studio environment. Remember to monitor both terminals for errors and progress.