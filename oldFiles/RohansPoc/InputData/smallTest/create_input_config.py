import json
import argparse
from pathlib import Path


def create_input_config(integrated_file: str, output_file: str):
    """
    Create an input config JSON with servers and prompts.
    Prompts are extracted from integrated.json.
    Servers are hardcoded (can be modified later).
    """
    # Load integrated data
    with open(integrated_file, 'r', encoding='utf-8') as f:
        integrated_data = json.load(f)
    
    # Extract prompts from integrated data
    prompts = [item["prompt"] for item in integrated_data]
    
    # Hardcoded servers section (user can modify this later)
    servers = [
        {
            "name": "gpt2-A",
            "url": "http://localhost:8001/v1/chat/completions",
            "output_file": "output_a.json"
        },
        {
            "name": "gpt2-B",
            "url": "http://localhost:8002/v1/chat/completions",
            "output_file": "output_b.json"
        }
    ]
    
    # Create the config structure
    config = {
        "servers": servers,
        "prompts": prompts
    }
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Created config with {len(servers)} servers and {len(prompts)} prompts")
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create input config from integrated data"
    )
    parser.add_argument(
        "--integrated",
        type=str,
        required=True,
        help="Path to integrated.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="input_config.json",
        help="Output config file path (default: input_config.json)"
    )
    
    args = parser.parse_args()
    
    create_input_config(args.integrated, args.output)


if __name__ == "__main__":
    main()
