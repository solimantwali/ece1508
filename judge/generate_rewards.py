import json
# I wrote this one
rewards_array = [[3, 5, 7, 10], [-3, 3, 5, 7], [-5, - 3, -1, 1], [-10, -5, -3, -1]]

# Chat wrote this one: 
# rewards_array = [[4, 6, 9, 10], [-1, 0, 2, 4], [-5, -3, -2, -6],[-8, -9, -9, -10]]

# Model types to process
MODEL_TYPES = ["baseline", "sft"]

def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def generate_rewards_for_model(model_type):
    """
    Generate scalar rewards for a model by comparing its judge labels
    with human judge labels.
    
    Args:
        model_type: "baseline" or "sft"
    """
    # Load model judged data
    model_file = f"oem_val_judged_{model_type}.json"
    human_file = "oem_val_judged_human.json"
    output_file = f"oem_val_{model_type}_with_rewards.json"
    
    print(f"\n{'='*60}")
    print(f"Processing: {model_type}")
    print(f"Model file: {model_file}")
    print(f"Human file: {human_file}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")
    
    model_data = load_json(model_file)
    human_data = load_json(human_file)
    
    # Create a dictionary for quick human data lookup by id
    human_dict = {}
    for entry in human_data:
        entry_id = entry["id"]
        judge_label = entry.get("judge_label", "")
        # Convert to int if it's a string
        if isinstance(judge_label, str) and judge_label.isdigit():
            judge_label = int(judge_label)
        human_dict[entry_id] = {
            "human_response": entry.get("human", ""),
            "judge_label_human": judge_label
        }
    
    # Process each model entry and calculate rewards
    rewards_calculated = 0
    output_data = []
    
    for entry in model_data:
        entry_id = entry["id"]
        
        # Get model judge label
        model_label = entry.get("judge_label", "")
        if isinstance(model_label, str) and model_label.isdigit():
            model_label = int(model_label)
        
        # Get corresponding human data
        human_info = human_dict.get(entry_id)
        
        if human_info is not None and isinstance(model_label, int) and isinstance(human_info["judge_label_human"], int):
            human_label = human_info["judge_label_human"]
            
            # Calculate reward using rewards_array[model_label][human_label]
            reward = rewards_array[model_label][human_label]
            
            # Create clean output entry with only relevant fields
            output_entry = {
                "id": entry_id,
                "sentence": entry.get("sentence", ""),
                "human_response": human_info["human_response"],
                "model_response": entry.get("base_reply", ""),
                "judge_label_human": human_label,
                "judge_label_model": model_label,
                "reward": reward
            }
            output_data.append(output_entry)
            rewards_calculated += 1
            
            if (rewards_calculated - 1) % 100 == 0:  # Print every 100th
                print(f"[{rewards_calculated}] ID {entry_id}: "
                      f"Model={model_label}, Human={human_label}, Reward={reward}")
        else:
            print(f"Warning: Could not calculate reward for ID {entry_id}")
            # Still add entry but with None reward
            output_entry = {
                "id": entry_id,
                "sentence": entry.get("sentence", ""),
                "human_response": human_info["human_response"] if human_info else "",
                "model_response": entry.get("base_reply", ""),
                "judge_label_human": human_info["judge_label_human"] if human_info else None,
                "judge_label_model": model_label if isinstance(model_label, int) else None,
                "reward": None
            }
            output_data.append(output_entry)
    
    # Save updated data
    save_json(output_file, output_data)
    
    print(f"\nProcessed {rewards_calculated} entries")
    print(f"Saved to: {output_file}")
    
    # Print reward statistics
    rewards = [e["reward"] for e in output_data if e.get("reward") is not None]
    if rewards:
        print(f"\nReward Statistics for {model_type}:")
        print(f"  Mean: {sum(rewards) / len(rewards):.2f}")
        print(f"  Min: {min(rewards)}")
        print(f"  Max: {max(rewards)}")
        print(f"  Total entries: {len(rewards)}")

# Process both model types
for model_type in MODEL_TYPES:
    generate_rewards_for_model(model_type)

print(f"\n{'='*60}")
print("All rewards generated successfully!")
print(f"{'='*60}")
