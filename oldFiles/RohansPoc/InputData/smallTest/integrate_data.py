import json
import argparse
from pathlib import Path


def integrate_data(pas_file: str, oeq_file: str, output_file: str):
    """
    Extract utterances from PAS and prompts from OEQ, combine them into a single JSON array.
    Each entry will have a "prompt" field.
    """
    # Load PAS data (utterances)
    with open(pas_file, 'r', encoding='utf-8') as f:
        pas_data = json.load(f)
    
    # Load OEQ data (prompts)
    with open(oeq_file, 'r', encoding='utf-8') as f:
        oeq_data = json.load(f)
    
    # Create integrated data array
    integrated = []
    
    # Add all utterances from PAS as prompts
    for item in pas_data:
        integrated.append({
            "prompt": item["utterance"]
        })
    
    # Add all prompts from OEQ
    for item in oeq_data:
        integrated.append({
            "prompt": item["prompt"]
        })
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(integrated, f, indent=2, ensure_ascii=False)
    
    print(f"Integrated {len(pas_data)} PAS utterances and {len(oeq_data)} OEQ prompts")
    print(f"Total entries: {len(integrated)}")
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate PAS utterances and OEQ prompts into a single JSON file"
    )
    parser.add_argument("--pas", type=str, required=True, help="Path to PAS JSON file")
    parser.add_argument("--oeq", type=str, required=True, help="Path to OEQ JSON file")
    parser.add_argument(
        "--output", 
        type=str, 
        default="integratedData.json",
        help="Output JSON file path (default: integratedData.json)"
    )
    
    args = parser.parse_args()
    
    integrate_data(args.pas, args.oeq, args.output)


if __name__ == "__main__":
    main()
