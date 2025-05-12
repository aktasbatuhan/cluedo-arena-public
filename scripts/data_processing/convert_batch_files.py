import json
import os
import glob
import argparse

def convert_file(input_file, output_file):
    """Convert a DRIA batch file from prompt format to messages format"""
    print(f"Converting {input_file} to {output_file}")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line.strip())
            
            # Check if this is already in the new format
            if "messages" in data.get("body", {}):
                fixed_lines.append(json.dumps(data))
                continue
                
            # Convert from old format (prompt field) to new format (messages array)
            if "prompt" in data.get("body", {}):
                prompt_content = data["body"]["prompt"]
                # Remove prompt field and add messages array
                data["body"]["messages"] = [{"role": "user", "content": prompt_content}]
                del data["body"]["prompt"]
                
            fixed_lines.append(json.dumps(data))
            
        except json.JSONDecodeError as e:
            print(f"  Error in line {line_num}: {e}")
            
    # Write the fixed data
    with open(output_file, 'w') as f:
        for line in fixed_lines:
            f.write(line + "\n")
            
    return len(fixed_lines)

def main():
    parser = argparse.ArgumentParser(description="Convert DRIA batch files from prompt to messages format")
    parser.add_argument("--input_dir", default="dria_batch_files", help="Directory containing batch JSONL files")
    parser.add_argument("--output_dir", default="dria_batch_files_fixed", help="Directory to output fixed files")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all JSONL files in the input directory
    jsonl_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {args.input_dir}")
        return
        
    # Process each file
    for input_file in jsonl_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(args.output_dir, filename)
        lines_fixed = convert_file(input_file, output_file)
        print(f"  Fixed {lines_fixed} entries in {filename}")
    
    print("Conversion complete")

if __name__ == "__main__":
    main() 