import json
import ast
import random
import argparse
from copy import deepcopy
import re

# --- Constants ---
CLUE_CONTEXT_PREFIX = (
    "You are an AI agent playing the board game Cluedo (also known as Clue), "
    "a deduction game where players try to determine the suspect, weapon, and room of a crime. "
    "Your task is to update your memory and deductions based on new events. "
    "Respond ONLY with a YAML object."
)

# This prefix is added if it's NOT the agent's first memory update.
NO_HAND_DEDUCTION_PREFIX = (
    "Important: For this memory update, do NOT list cards that are already in your hand as 'newlyDeducedCards'. "
    "Focus ONLY on cards that you have deduced for the first time based on the game events from *this* turn "
    "or by logical inference from other players' actions or revealed cards. "
    "Cards from your own hand should already be part of your permanent memory and "
    "not re-stated as 'newly deduced' in this update. "
)

FIRST_MEMORY_UPDATE_MARKER = "(No previous memory summary)"

# --- Helper Functions (adapted from prepare_predibase_data.py) ---

def split_prompt_sections(prompt: str):
    """
    Splits the prompt into (instructions_and_format, context_sections).
    """
    sections = prompt.strip().split('\n\n')
    context_start_index = 0
    for i, sec in enumerate(sections):
        # Heuristics to find where the game context (knowledge, events) starts
        if (sec.strip().lower().startswith('your current knowledge:') or
            sec.strip().lower().startswith('events from this turn:') or
            sec.strip().lower().startswith('cards in my hand:') or
            sec.strip().lower().startswith('known eliminated cards:')):
            context_start_index = i
            break
    else:
        # If no clear context headers found, assume first major block is instructions/format, rest is context
        # This might need adjustment based on typical prompt structure.
        # For now, let's assume at least one block for instructions/format if split is possible.
        context_start_index = 1 if len(sections) > 1 else len(sections)

    instructions_and_format_sections = sections[:context_start_index]
    context_sections = sections[context_start_index:]
    return instructions_and_format_sections, context_sections

def shuffle_prompt_context_sections(prompt: str) -> str:
    """
    Shuffle only the context/game sections (e.g., 'Your current knowledge:', 'Events from THIS turn:'),
    keeping instructions and YAML format guidance at the top.
    """
    if not prompt:
        return ""
    
    # Split the prompt more carefully
    # Top part: CLUE_CONTEXT_PREFIX + NO_HAND_DEDUCTION_PREFIX (if applicable) + initial instructions from original prompt
    # Original prompt's context: "Your current knowledge:", "Events from THIS turn:"
    # Bottom part: YAML format instructions ("Respond ONLY with a YAML object...")

    # We need to isolate the parts of the *original_prompt_text* that are actual context.
    # The prefixes are added *around* the original prompt or parts of it.

    # Simpler approach: the original `shuffle_prompt_sections` from prepare_predibase_data.py
    # operates on the LLM's input prompt which *already contains* context blocks.
    # We will apply it to the `current_prompt_text` before adding our prefixes.

    # This function will be called with the original prompt text from the memory_update entry.
    instructions_and_format, context_sections = split_prompt_sections(prompt)
    
    if not context_sections: # If split didn't find distinct context, don't shuffle
        return prompt

    random.shuffle(context_sections)
    return '\n\n'.join(instructions_and_format + context_sections)


def main(args):
    seen_prompts_and_gt = set()
    final_output_records = []
    skipped_memory_updates = 0
    memory_updates_processed = 0
    deduction_comparisons_found = 0
    kept_after_filtering = 0

    try:
        with open(args.input, 'r', encoding='utf-8-sig') as fin:
            all_interactions = json.load(fin)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file {args.input}")
        return

    if not isinstance(all_interactions, list):
        print(f"Error: Input file {args.input} does not contain a JSON list.")
        return

    num_total_interactions = len(all_interactions)

    # --- 3-card deduction filter setup ---
    filter3card_prob = args.filter3card_deductions

    processing_indices = []
    if args.fetch_recent and args.fetch_recent > 0:
        # Determine the actual indices of the 'X' most recent memory_update items
        selected_indices = []
        count = 0
        for i in range(num_total_interactions - 1, -1, -1): # Iterate backwards
            interaction_candidate = all_interactions[i]
            if isinstance(interaction_candidate, dict) and interaction_candidate.get("type") == "memory_update":
                selected_indices.append(i) # Add original index
                count += 1
                if count >= args.fetch_recent:
                    break
        processing_indices = sorted(selected_indices) # Sort to process in chronological order for GT lookup
        print(f"Fetching {len(processing_indices)} most recent memory_update entries based on --fetch-recent {args.fetch_recent}.")
    else:
        processing_indices = range(num_total_interactions) # Process all

    for i in processing_indices:
        interaction = all_interactions[i]

        if not isinstance(interaction, dict):
            print(f"Warning: Skipping non-dictionary item in input list: {interaction}")
            continue

        interaction_type = interaction.get("type")
        if interaction_type != "memory_update":
            continue
        
        memory_updates_processed += 1
        
        current_agent_name = interaction.get("agent")
        input_data = interaction.get("input", {})
        if not isinstance(input_data, dict):
            print(f"Warning: Skipping memory_update for agent {current_agent_name} due to invalid 'input' field. Timestamp: {interaction.get('timestamp', 'N/A')}")
            skipped_memory_updates += 1
            continue
            
        current_prompt_text = input_data.get("prompt")
        if not isinstance(current_prompt_text, str) or not current_prompt_text:
            print(f"Warning: Skipping memory_update for agent {current_agent_name} due to missing or invalid 'prompt'. Timestamp: {interaction.get('timestamp', 'N/A')}")
            skipped_memory_updates += 1
            continue

        # Look for the corresponding deduction_comparison
        ground_truth_list = None
        if i + 1 < num_total_interactions:
            next_interaction = all_interactions[i+1]
            if (isinstance(next_interaction, dict) and
                next_interaction.get("type") == "deduction_comparison" and
                next_interaction.get("agent") == current_agent_name):
                
                gt_data = next_interaction.get("groundTruthDeductions")
                if isinstance(gt_data, list):
                    ground_truth_list = [str(item) for item in gt_data if isinstance(item, (str, int, float))]
                    deduction_comparisons_found +=1
                else:
                    print(f"Warning: 'groundTruthDeductions' in deduction_comparison for agent {current_agent_name} is not a list or missing. Timestamp: {next_interaction.get('timestamp', 'N/A')}")
                    # ground_truth_list remains None
            else:
                 print(f"Debug: memory_update for {current_agent_name} at index {i} not immediately followed by its deduction_comparison. Next type: {next_interaction.get('type')}, agent: {next_interaction.get('agent')}")


        if ground_truth_list is None:
            print(f"Warning: Could not find valid groundTruthDeductions for memory_update of agent {current_agent_name}. Timestamp: {interaction.get('timestamp', 'N/A')}")
            skipped_memory_updates += 1
            continue

        # Filter based on ground_truth_list length
        if not ground_truth_list and random.random() >= args.keep_empty_prob:
            skipped_memory_updates += 1
            continue
        # --- Filter 3-card ground truth deductions if requested ---
        if filter3card_prob is not None and len(ground_truth_list) == 3:
            if random.random() >= filter3card_prob:
                skipped_memory_updates += 1
                continue
        
        # Determine if it's the first memory update
        is_first_update = FIRST_MEMORY_UPDATE_MARKER in current_prompt_text

        # Shuffle original prompt context if enabled
        processed_prompt_text = current_prompt_text
        if args.shuffle_prompt_context:
            processed_prompt_text = shuffle_prompt_context_sections(current_prompt_text)

        # Construct the final prompt
        final_prompt_parts = [CLUE_CONTEXT_PREFIX]
        if not is_first_update and not args.no_conditional_hand_prefix:
            final_prompt_parts.append(NO_HAND_DEDUCTION_PREFIX)
        final_prompt_parts.append(processed_prompt_text)
        final_prompt = "\n\n".join(final_prompt_parts)
        
        # Deduplicate
        # Sort ground_truth_list for consistent dedup key
        dedup_key_gt_str = json.dumps(sorted(ground_truth_list))
        dedup_key = final_prompt + "|||" + dedup_key_gt_str
        
        if dedup_key not in seen_prompts_and_gt:
            seen_prompts_and_gt.add(dedup_key)
            final_output_records.append({
                "prompt": final_prompt,
                "ground_truth_deductions": json.dumps(ground_truth_list) # Save as JSON string
            })
            kept_after_filtering +=1
        else:
            # This path also means it was skipped due to deduplication after other checks passed
            pass # Not necessarily an error, just a duplicate

    print(f"\n--- Processing Summary ---")
    print(f"Total interactions in input file: {num_total_interactions}")
    print(f"Memory update entries initially found: {memory_updates_processed}")
    print(f"Corresponding deduction_comparison entries successfully processed: {deduction_comparisons_found}")
    print(f"Memory updates skipped (missing prompt, GT, or filtered out by empty prob): {skipped_memory_updates}")
    print(f"Records kept after all filtering and deduplication: {len(final_output_records)}")
    
    duplicates_skipped = deduction_comparisons_found - kept_after_filtering - skipped_memory_updates
    # The above calculation for duplicates might be off, more simply:
    # Total unique prompts added to seen_prompts_and_gt is len(final_output_records)
    # Total that could have been added if no duplicates = deduction_comparisons_found - those skipped by empty_prob.
    # This stat is tricky. Let's just report what's written.

    try:
        with open(args.output, "w", encoding='utf-8') as fout:
            for record in final_output_records:
                print(json.dumps(record, ensure_ascii=False), file=fout)
        print(f"Successfully wrote {len(final_output_records)} records to {args.output}")
    except IOError:
        print(f"Error: Could not write to output file {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Cluedo memory update training data from llm_interactions.json.")
    parser.add_argument("--input", required=True, help="Input llm_interactions.json file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path for Predibase training")
    parser.add_argument("--keep-empty-prob", type=float, default=0.05, 
                        help="Fraction of memory_update examples with empty ground_truth_deductions to keep (default: 0.05)")
    parser.add_argument("--shuffle-prompt-context", action="store_true", 
                        help="Enable shuffling of context sections (e.g., 'Your current knowledge', 'Events from THIS turn') within the prompt.")
    parser.add_argument("--no-conditional-hand-prefix", action="store_true",
                        help="Disable the conditional prefix that instructs the model not to deduce hand cards on non-first turns.")
    parser.add_argument("--filter3card-deductions", type=float, default=None,
                        help="Fraction of entries with exactly 3 ground_truth_deductions to keep (default: keep all)")
    parser.add_argument("--fetch-recent", type=int, default=None,
                        help="Fetch only the most recent X memory_update entries to process before other filters.")
    
    args = parser.parse_args()
    # --- 3-card deduction filter setup ---
    filter3card_prob = args.filter3card_deductions
    main(args) 