"""
splitter.py

Splits a JSON file containing training data into multiple smaller files.
The split is based on configurable boundaries (e.g., first 1k samples, second 1k, remaining).

Usage:
    python splitter.py
"""

import json

# ============================================================
# USER CONFIGURATION
# ============================================================

# Input file to split
INPUT_FILE = "oem_train_baseline.json"

# Output file prefix (will be appended with split identifiers)
OUTPUT_PREFIX = "oem_train_baseline"

# Split boundaries: list of (start_index, end_index, suffix) tuples
# start_index is inclusive, end_index is exclusive (None means to the end)
SPLITS = [
    (0, 1000, "first_1k"),      # First 1000 samples (0-999)
    (1000, 2000, "1k_2k"),      # Second 1000 samples (1000-1999)
    (2000, None, "2k_plus"),    # Remaining samples (2000+)
]

# ============================================================
# SCRIPT LOGIC - DO NOT MODIFY BELOW
# ============================================================

def load_json(filepath):
    """Load JSON file and return data."""
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    """Save data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def split_dataset():
    """
    Split a large JSON dataset into smaller chunks based on configured boundaries.
    """
    print(f"\n{'='*60}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output prefix: {OUTPUT_PREFIX}")
    print(f"Number of splits: {len(SPLITS)}")
    print(f"{'='*60}\n")
    
    # Load the full dataset
    print("Loading dataset...")
    full_data = load_json(INPUT_FILE)
    total_samples = len(full_data)
    print(f"Total samples in dataset: {total_samples}\n")
    
    # Process each split
    for start_idx, end_idx, suffix in SPLITS:
        # Determine actual end index
        actual_end = end_idx if end_idx is not None else total_samples
        
        # Extract subset
        subset = full_data[start_idx:actual_end]
        num_samples = len(subset)
        
        # Generate output filename
        output_file = f"{OUTPUT_PREFIX}_{suffix}.json"
        
        # Save subset
        save_json(output_file, subset)
        
        print(f"✓ Created {output_file}")
        print(f"  Range: [{start_idx}:{actual_end}]")
        print(f"  Samples: {num_samples}\n")
    
    print(f"{'='*60}")
    print("Split completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    split_dataset()
