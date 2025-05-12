import os
import argparse
from predibase import Predibase

# --- Configuration ---
PREDIBASE_API_KEY = os.getenv("PREDIBASE_API_KEY")
if not PREDIBASE_API_KEY:
    raise ValueError("PREDIBASE_API_KEY environment variable not set.")

# --- Initialize Predibase Client ---
try:
    pb = Predibase(api_token=PREDIBASE_API_KEY)
    print("Predibase client initialized successfully.")
except Exception as e:
    print(f"Error initializing Predibase client: {e}")
    print("Ensure PREDIBASE_API_KEY environment variable is set correctly.")
    exit(1)

def prompt_model(deployment_name: str, adapter_id_str: str, prompt_text: str, max_new_tokens: int = 256):
    """
    Sends a prompt to a deployed Predibase model with a specific adapter and prints the response.

    Args:
        deployment_name: The name of the base model deployment (e.g., 'qwen2-5-7b-instruct').
        adapter_id_str: The full adapter identifier (e.g., 'your_repo/name/version').
        prompt_text: The text to prompt the model with.
        max_new_tokens: The maximum number of new tokens to generate.
    """
    try:
        print(f"\nAttempting to get client for deployment: '{deployment_name}'")
        # Get a client for the serverless deployment of the base model
        # The adapter is specified during the generate call.
        client = pb.deployments.client(deployment_name)
        print(f"Successfully got client for deployment: '{deployment_name}'")

        print(f"Sending prompt to adapter '{adapter_id_str}'...")
        print(f"Prompt: {prompt_text}")

        response = client.generate(
            prompt_text,
            adapter_id=adapter_id_str,
            max_new_tokens=max_new_tokens
        )
        
        print("\n--- Model Response ---")
        print(response.generated_text)
        print("----------------------")

    except Exception as e:
        print(f"\n--- Error during prompting --- ")
        print(e)
        print("Please ensure:")
        print(f"1. The deployment '{deployment_name}' exists and is active.")
        print(f"2. The adapter '{adapter_id_str}' exists and is compatible.")
        print("3. Your API key has the necessary permissions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt a Predibase model with a specific adapter.")
    parser.add_argument("--deployment", default="qwen2-5-7b-instruct", 
                        help="Name of the base model deployment in Predibase (default: qwen2-5-7b-instruct).")
    parser.add_argument("--adapter", required=True, 
                        help="Full adapter ID (e.g., 'your_repo/adapter_name/version' or 'adapter_name/version').")
    parser.add_argument("--prompt", required=True, help="The prompt text to send to the model.")
    parser.add_argument("--max-tokens", type=int, default=256, 
                        help="Maximum new tokens to generate (default: 256).")
    
    args = parser.parse_args()

    # Construct the full adapter ID if it doesn't contain a slash (assuming it's in the default repo)
    # This might need adjustment based on how your adapters are named and stored.
    # For now, we assume the user provides the full path if it's not in the default repo.
    adapter_id = args.adapter
    # Example: if user provides 'my_adapter/1' and your repo is 'my_org', it should be 'my_org/my_adapter/1'
    # The pb.deployments.client(deployment_name).generate(..., adapter_id=...) expects the full path.

    prompt_model(args.deployment, adapter_id, args.prompt, max_new_tokens=args.max_tokens) 