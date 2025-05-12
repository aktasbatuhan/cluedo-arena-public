import json
import random
import argparse

def main(input_path: str, output_path: str, sample_size: int):
    """Reads a JSONL file, randomly samples a specified number of lines, and writes to a new JSONL file."""
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            all_lines = infile.readlines()
        
        if len(all_lines) < sample_size:
            print(f"Warning: Requested sample size ({sample_size}) is larger than the number of lines in the file ({len(all_lines)}).")
            print(f"The output file will contain all lines from the input file.")
            sampled_lines = all_lines
        else:
            sampled_lines = random.sample(all_lines, sample_size)
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in sampled_lines:
                outfile.write(line) # Assumes lines already have newlines
        
        print(f"Successfully sampled {len(sampled_lines)} lines from '{input_path}' and saved to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly sample lines from a JSONL file.")
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path for the sampled data.")
    parser.add_argument("--sample-size", type=int, required=True, help="Number of lines to randomly sample.")
    args = parser.parse_args()
    
    if args.sample_size <= 0:
        print("Error: Sample size must be a positive integer.")
    else:
        main(args.input, args.output, args.sample_size) 