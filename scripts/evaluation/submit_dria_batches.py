import os
import requests
import json
import argparse
import glob
import time # Added for delays

# Updated base URL from the new example
DRIA_API_BASE_URL = "https://mainnet.dkn.dria.co/api/v0"
GET_UPLOAD_URL_ENDPOINT = f"{DRIA_API_BASE_URL}/file/get_upload_url"  # Step 1 endpoint
COMPLETE_UPLOAD_ENDPOINT = f"{DRIA_API_BASE_URL}/batch/complete_upload" # Step 3 endpoint

DRIA_API_KEY = "dria_91f81e8e17313038e378b27fbe6f6841hayatbizi61kenara61616161BATUHAN"
DELAY_BETWEEN_FILES = 5  # Seconds to wait between processing each file
LOOP_DELAY = 10 # Seconds to wait between submission loops

def get_upload_url_and_id(api_key, filename_for_logging): # filename is mostly for logging here
    """Step 1: Get upload URL and file ID from Dria."""
    headers = {
        "x-api-key": api_key
    }
    print(f"  Requesting upload URL for {filename_for_logging} from {GET_UPLOAD_URL_ENDPOINT}...")
    try:
        response = requests.get(GET_UPLOAD_URL_ENDPOINT, headers=headers, timeout=60) # Increased timeout slightly
        response.raise_for_status()
        data = response.json()
        
        upload_url = data.get("url")
        file_id = data.get("id")

        if not upload_url or not file_id:
            print(f"  Error: 'url' or 'id' not found in response: {data}")
            return None, None
        
        print(f"  Successfully obtained upload URL. Dria File ID: {file_id}")
        return upload_url, file_id
    except requests.exceptions.RequestException as e:
        print(f"  Error getting upload URL for {filename_for_logging}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response text: {e.response.text}")
        return None, None
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON response from get_upload_url endpoint. Response text: {response.text if 'response' in locals() else 'Unknown response'}")
        return None, None

def upload_file_to_s3(upload_url, filepath):
    """Step 2: Upload your file to the provided S3 URL."""
    print(f"  Uploading {filepath} to S3 URL...")
    try:
        file_size = str(os.path.getsize(filepath))
        with open(filepath, 'rb') as f:
            headers = {
                "Content-Type": "binary/octet-stream",
                "Content-Length": file_size,
            }
            response = requests.put(upload_url, data=f, headers=headers, timeout=300)
            response.raise_for_status()
        print(f"  File {filepath} uploaded successfully to S3.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Error uploading file {filepath} to S3: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response text: {e.response.text}")
        return False
    except FileNotFoundError:
        print(f"  Error: File not found at {filepath} for S3 upload.")
        return False
    except Exception as e:
        print(f"  An unexpected error occurred during S3 file upload: {e}")
        return False

def complete_dria_upload(api_key, file_id, filename_for_logging):
    """Step 3: Notify Dria that the upload is complete."""
    if not file_id:
        print(f"  Skipping complete_dria_upload for {filename_for_logging} as file_id was not available.")
        return False

    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "id": file_id 
    }
    print(f"  Notifying Dria of completed upload for file ID: {file_id} (filename: {filename_for_logging}) at {COMPLETE_UPLOAD_ENDPOINT}...")
    try:
        response = requests.post(COMPLETE_UPLOAD_ENDPOINT, headers=headers, json=payload, timeout=60) # Increased timeout slightly
        response.raise_for_status()
        print(f"  Upload completion for {filename_for_logging} (file ID: {file_id}) notified successfully.")
        # print(f"  Dria response: {response.json()}") # Keep this commented unless debugging needed for success cases
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Error notifying Dria of completed upload for {filename_for_logging} (file ID: {file_id}): {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response text: {e.response.text}")
        return False
    except json.JSONDecodeError:
        print(f"  Error: Could not decode JSON response from complete_dria_upload endpoint. Response text: {response.text if 'response' in locals() else 'Unknown response'}")
        return False

def main(args):
    # Hardcoded API key is used directly from global scope
    if not DRIA_API_KEY:
        print("Error: DRIA_API_KEY is not set in the script.")
        return

    batch_files_pattern = os.path.join(args.input_dir, "dria_batch_*.jsonl")
    batch_files = glob.glob(batch_files_pattern)

    if not batch_files:
        print(f"No batch files found matching pattern '{batch_files_pattern}'.")
        print(f"Please ensure files generated by 'prepare_dria_batch.py' are in '{args.input_dir}'.")
        return

    print(f"Found {len(batch_files)} batch files to submit to Dria using base URL: {DRIA_API_BASE_URL}")
    print(f"Will run the submission cycle {args.submit_count} time(s).")

    total_successful_submissions = 0
    total_failed_submissions = 0

    for loop_num in range(1, args.submit_count + 1):
        print(f"\n--- Starting Submission Loop {loop_num}/{args.submit_count} ---")
        loop_successful_submissions = 0
        loop_failed_submissions = 0
        loop_obtained_file_ids = [] # Reset for each loop

        for i, filepath in enumerate(batch_files):
            filename = os.path.basename(filepath)
            print(f"\nProcessing file {i+1}/{len(batch_files)}: {filename} (Loop {loop_num})")

            # Step 1: Get upload URL and file ID
            s3_upload_url, dria_file_id = get_upload_url_and_id(DRIA_API_KEY, filename)
            if not s3_upload_url or not dria_file_id:
                print(f"  Failed to get S3 upload URL or Dria file ID for {filename}. Skipping this file.")
                loop_failed_submissions += 1
                if i < len(batch_files) - 1: # Don't sleep after the last file in the inner loop
                    print(f"  Waiting for {DELAY_BETWEEN_FILES} seconds before next file...")
                    time.sleep(DELAY_BETWEEN_FILES)
                continue
            
            loop_obtained_file_ids.append({"filename": filename, "dria_file_id": dria_file_id})

            # Step 2: Upload file to S3
            upload_successful = upload_file_to_s3(s3_upload_url, filepath)
            if not upload_successful:
                print(f"  Failed to upload {filename} to S3. Skipping Dria completion step.")
                loop_failed_submissions += 1
                if i < len(batch_files) - 1:
                    print(f"  Waiting for {DELAY_BETWEEN_FILES} seconds before next file...")
                    time.sleep(DELAY_BETWEEN_FILES)
                continue

            # Step 3: Notify Dria of completion
            # Small delay before completing, as S3 upload might take a moment to fully register
            print("  Short pause (2s) before notifying Dria of completion...")
            time.sleep(2)
            completion_notified = complete_dria_upload(DRIA_API_KEY, dria_file_id, filename)
            
            if completion_notified:
                print(f"  Successfully submitted and completed {filename} with Dria (File ID: {dria_file_id}).")
                loop_successful_submissions +=1
            else:
                print(f"  S3 upload of {filename} was successful, but Dria completion notification failed.")
                print("  You may need to check Dria's web interface for batch status or attempt completion manually for this File ID.")
                loop_failed_submissions +=1
            
            if i < len(batch_files) - 1: # Don't sleep after the last file in the inner loop
                print(f"  Waiting for {DELAY_BETWEEN_FILES} seconds before next file...")
                time.sleep(DELAY_BETWEEN_FILES)

        # --- Summary for the current loop ---
        print(f"\n--- Loop {loop_num}/{args.submit_count} Summary ---")
        print(f"Successfully submitted AND completed files in this loop: {loop_successful_submissions}")
        print(f"Failed or incomplete submissions in this loop: {loop_failed_submissions}")
        print(f"\n--- Obtained Dria File IDs in Loop {loop_num} (from Step 1) ---")
        if loop_obtained_file_ids:
            for item in loop_obtained_file_ids:
                print(f"  Filename: {item['filename']}, Dria File ID: {item['dria_file_id']}")
        else:
            print("  No Dria File IDs were obtained in this loop (Step 1 failed for all files).")

        total_successful_submissions += loop_successful_submissions
        total_failed_submissions += loop_failed_submissions
        
        # Delay before next loop iteration
        if loop_num < args.submit_count:
            print(f"\n--- End of Loop {loop_num}. Waiting for {LOOP_DELAY} seconds before starting Loop {loop_num + 1}... ---")
            time.sleep(LOOP_DELAY)
            
    print("\n=== Overall Submission Summary Across All Loops ===")
    print(f"Total successfully submitted AND completed files across {args.submit_count} loop(s): {total_successful_submissions}")
    print(f"Total failed or incomplete submissions across {args.submit_count} loop(s): {total_failed_submissions}")
        
    print("\nReminder: Batch processing is asynchronous. Check Dria's web interface for job status and results after successful completion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit pre-generated .jsonl batch files to the Dria Batch Inference API, optionally looping the process.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the .jsonl files generated by prepare_dria_batch.py (e.g., ./dria_batch_files_fixed).")
    parser.add_argument("--submit_count", type=int, default=1, help="Number of times to repeat the entire submission process for all found files. Defaults to 1.")
    
    args = parser.parse_args()
    main(args) 