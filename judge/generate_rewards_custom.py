import json

# Rewards matrix
rewards_array = [[3, 5, 7, 10], [-3, 3, 5, 7], [-5, -3, -1, 1], [-10, -5, -3, -1]]

# ========================================
# CONFIGURATION - EDIT THESE
# ========================================
HUMAN_FILE = "oem_train_2k_plus_judged_human.json"
MODEL_FILE = "oem_train_2k_plus_judged_base_reply.json"  # Change to your model file
OUTPUT_FILE = "oem_train_2k_plus_with_rewards.json"  # Change output name as needed
# ========================================

def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def generate_rewards():
    """
    Generate scalar rewards by comparing model judge labels with human judge labels.
    """
    print(f"\n{'='*60}")
    print(f"Human file: {HUMAN_FILE}")
    print(f"Model file: {MODEL_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"{'='*60}\n")
    
    model_data = load_json(MODEL_FILE)
    human_data = load_json(HUMAN_FILE)
    
    # Create a dictionary for quick human data lookup by id
    human_dict = {}
    for entry in human_data:
        entry_id = entry["id"]
        judge_label = entry.get("judge_label_human", "")
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
        model_label = entry.get("judge_label_base_reply", "")
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
    save_json(OUTPUT_FILE, output_data)
    
    print(f"\nProcessed {rewards_calculated} entries")
    print(f"Saved to: {OUTPUT_FILE}")
    
    # Print reward statistics
    rewards = [e["reward"] for e in output_data if e.get("reward") is not None]
    if rewards:
        print(f"\nReward Statistics:")
        print(f"  Mean: {sum(rewards) / len(rewards):.2f}")
        print(f"  Min: {min(rewards)}")
        print(f"  Max: {max(rewards)}")
        print(f"  Total entries: {len(rewards)}")

if __name__ == "__main__":
    generate_rewards()
    print(f"\n{'='*60}")
    print("Rewards generated successfully!")
    print(f"{'='*60}")
