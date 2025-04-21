import argparse
import asyncio
import logging
from collections import deque, defaultdict
from uuid import uuid4

import art
import openai
from flask import Flask, jsonify, request

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Globals ---
# Temporary storage for trajectory parts (request_id -> {user_message, assistant_choice})
trajectory_parts_store = {} 
# Batch of completed trajectories to send to ART
training_batch = deque()
# TODO: Add validation batch if needed
# validation_batch = deque() 

# ART Model (initialized later)
art_model = None
openai_client = None

# Flask App
app = Flask(__name__)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="ART Wrapper for Clue Game RL Training")
    parser.add_argument("--project", type=str, default="cluedo-art", help="ART project name")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="Base model for ART training")
    parser.add_argument("--port", type=int, default=5001, help="Port for the wrapper API server")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for sending trajectories to ART training")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for ART training")
    # Add other ART config arguments as needed (e.g., server address if not local)
    return parser.parse_args()

# --- ART Initialization ---
async def initialize_art(args):
    global art_model, openai_client
    logger.info(f"Initializing ART model: {args.base_model} for project: {args.project}")
    try:
        # Create local API endpoint (remove explicit host)
        local_api = art.LocalAPI() 
        
        art_model = art.TrainableModel(
            name="cluedo-agent-001", # Example name, could be configurable
            project=args.project,
            base_model=args.base_model,
            # Add any specific internal_config if needed
            # For VRAM optimization, you may need to adjust this
            _internal_config={"init_args": {"gpu_memory_utilization": 0.7}} 
        )
        
        # Register with LocalAPI created above (NOT await model.register(art.LocalAPI()))
        logger.info(f"Registering model with the LocalAPI server...")
        await art_model.register(local_api)
        
        # Note: LocalAPI() runs its own server automatically as a background task
        # when initialized, so we don't need to start it separately.
        
        openai_client = art_model.openai_client()
        logger.info("ART model registered and OpenAI client obtained.")
        
        # Log initial step
        step = await art_model.get_step()
        logger.info(f"Initial ART training step: {step}")
        
    except Exception as e:
        logger.error(f"Failed to initialize ART: {e}", exc_info=True)
        raise

# --- Flask API Endpoints ---

@app.route('/llm_request', methods=['POST'])
async def handle_llm_request():
    global trajectory_parts_store, openai_client, art_model
    if not openai_client or not art_model:
        return jsonify({"error": "ART components not initialized"}), 500
        
    data = request.get_json()
    if not data or 'prompt' not in data or 'agent_name' not in data or 'turn_number' not in data:
        logger.warning(f"Received invalid LLM request data: {data}")
        return jsonify({"error": "Missing required fields: prompt, agent_name, turn_number"}), 400

    prompt = data['prompt']
    agent_name = data['agent_name']
    turn_number = data['turn_number']
    # task_type = data.get('task_type', 'unknown') # Optional: use if needed

    request_id = str(uuid4())
    user_message = {"role": "user", "content": prompt}
    messages = [user_message]

    try:
        logger.info(f"Received LLM request {request_id} for {agent_name} Turn {turn_number}. Calling ART client.")
        chat_completion = await openai_client.chat.completions.create(
            messages=messages, 
            model=art_model.name # Use the ART model registered name
            # Add other parameters like temperature if needed
        )
        choice = chat_completion.choices[0]
        assistant_choice = choice # Store the whole choice object
        assistant_content = choice.message.content

        # Store the parts needed for the trajectory
        trajectory_parts_store[request_id] = {
            "user_message": user_message,
            "assistant_choice": assistant_choice
        }
        logger.info(f"Stored trajectory parts for request {request_id}")

        return jsonify({"request_id": request_id, "content": assistant_content})

    except openai.APIError as e:
        logger.error(f"OpenAI API error during LLM request {request_id}: {e}")
        return jsonify({"error": f"OpenAI API Error: {e}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error during LLM request {request_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error handling LLM request"}), 500

@app.route('/log_trajectory', methods=['POST'])
async def handle_log_trajectory():
    global training_batch, trajectory_parts_store
    
    data = request.get_json()
    if not data or 'request_id' not in data or 'reward' not in data:
        logger.warning(f"Received invalid log trajectory data: {data}")
        return jsonify({"error": "Missing required fields: request_id, reward"}), 400

    request_id = data['request_id']
    reward = float(data['reward'])
    metrics = data.get('metrics', {}) # Optional metrics

    if request_id not in trajectory_parts_store:
        logger.error(f"Cannot log trajectory: request_id {request_id} not found in store. Maybe already processed or invalid?")
        return jsonify({"error": "request_id not found or already processed"}), 404

    try:
        parts = trajectory_parts_store.pop(request_id) # Retrieve and remove
        user_message = parts['user_message']
        assistant_choice = parts['assistant_choice']

        trajectory = art.Trajectory(
            messages_and_choices=[user_message, assistant_choice],
            reward=reward,
            metrics=metrics
        )
        
        # TODO: Distinguish between training and validation trajectories if needed
        # For now, add all to training batch
        training_batch.append(trajectory)
        logger.info(f"Added trajectory for request {request_id} to training batch. Batch size: {len(training_batch)}")

        return jsonify({"status": "trajectory logged"}), 200

    except KeyError: # Double check in case of race condition (though pop should be atomic enough)
        logger.error(f"Race condition or duplicate request? request_id {request_id} not found during pop.")
        return jsonify({"error": "request_id not found during processing"}), 404
    except Exception as e:
        logger.error(f"Error processing trajectory log for {request_id}: {e}", exc_info=True)
        # Re-add parts to store? Or discard?
        # trajectory_parts_store[request_id] = parts # Example: Put it back if processing fails
        return jsonify({"error": "Internal server error logging trajectory"}), 500

# --- Training Logic ---
async def train_batch_periodically(args):
    global training_batch
    while True:
        await asyncio.sleep(5) # Check every 5 seconds
        if len(training_batch) >= args.batch_size:
            logger.info(f"Training batch size {args.batch_size} reached. Starting training.")
            batch_to_train = list(training_batch)
            training_batch.clear()
            
            try:
                # Log validation data if needed before training
                # if validation_batch:
                #    await art_model.log(list(validation_batch))
                #    validation_batch.clear()

                await art_model.train(
                    batch_to_train,
                    config=art.TrainConfig(learning_rate=args.learning_rate),
                )
                step = await art_model.get_step()
                logger.info(f"Training step {step} completed.")
                # Optionally delete old checkpoints
                # await art_model.delete_checkpoints() 
            except Exception as e:
                logger.error(f"ART training failed: {e}", exc_info=True)
                # Decide how to handle failed batch (retry, discard, log?)
                # For now, just log the error. Could re-add to deque?


# --- Main Execution ---
async def main():
    args = parse_args()
    
    logger.info("Initializing ART components (combines ART server and wrapper API)...")
    await initialize_art(args)
    
    # Start the background training task
    asyncio.create_task(train_batch_periodically(args))
    
    # Now we can start Flask server
    logger.info(f"Starting Flask API server on port {args.port}")
    
    # Use Hypercorn to run the Flask app within the async context
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    
    config = Config()
    config.bind = [f"0.0.0.0:{args.port}"]
    # Add any other hypercorn config settings as needed
    
    # Run the Flask app with Hypercorn in the same event loop
    await serve(app, config)


if __name__ == '__main__':
    # Python 3.7+
    asyncio.run(main()) 