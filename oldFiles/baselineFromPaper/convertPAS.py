import csv
import json
import argparse
from pathlib import Path


def convert_csv_to_json(csv_file: str, output_json: str):
    """
    Extract prompt from 'utterance' column and response from 'response' column.
    Save as JSON array with prompt and response fields.
    Uses sentence_id to enforce no duplicates.
    """
    data = []
    seen_ids = set()
    duplicates_skipped = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if required columns exist
        if 'utterance' not in reader.fieldnames or 'response' not in reader.fieldnames:
            raise ValueError(f"Missing required columns. Available columns: {reader.fieldnames}")
        
        if 'sentence_id' not in reader.fieldnames:
            raise ValueError(f"Missing 'sentence_id' column. Available columns: {reader.fieldnames}")
        
        for row in reader:
            sentence_id = row['sentence_id']
            
            # Skip duplicates
            if sentence_id in seen_ids:
                duplicates_skipped += 1
                continue
            
            seen_ids.add(sentence_id)
            
            obj = {
                "prompt": row['utterance'],
                "response": row['response']
            }
            data.append(obj)
    
    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} unique rows from {csv_file}")
    if duplicates_skipped > 0:
        print(f"Skipped {duplicates_skipped} duplicate entries")
    print(f"Saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert llamaPAS.csv to JSON format"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="llamaPAS.csv",
        help="Input CSV file path (default: llamaPAS.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llamaPAS.json",
        help="Output JSON file path (default: llamaPAS.json)"
    )
    
    args = parser.parse_args()
    
    convert_csv_to_json(args.csv, args.output)


if __name__ == "__main__":
    main()
