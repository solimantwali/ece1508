import csv
import json
import argparse
from pathlib import Path


def convert_csv_to_json(csv_file: str, output_json: str):
    """
    Extract prompt from 'sentence' column and response from 'Llama-8B' column.
    Save as JSON array with prompt and response fields.
    """
    data = []
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        # Check if required columns exist
        if 'sentence' not in reader.fieldnames or 'Llama-8B' not in reader.fieldnames:
            raise ValueError(f"Missing required columns. Available columns: {reader.fieldnames}")
        
        for row in reader:
            obj = {
                "prompt": row['sentence'],
                "response": row['Llama-8B']
            }
            data.append(obj)
    
    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(data)} rows from {csv_file}")
    print(f"Saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert llama8BOEQ.csv to JSON format"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="llama8BOEQ.csv",
        help="Input CSV file path (default: llama8BOEQ.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llama8BOEQ.json",
        help="Output JSON file path (default: llama8BOEQ.json)"
    )
    
    args = parser.parse_args()
    
    convert_csv_to_json(args.csv, args.output)


if __name__ == "__main__":
    main()
