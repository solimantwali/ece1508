import json
import argparse


def trim_prompt(prompt: str, max_tokens: int = 600) -> str:
    """
    Trim a prompt to approximately max_tokens using character estimation.
    Rough estimate: 1 token ≈ 4 characters
    """
    max_chars = max_tokens * 4  # ~2400 characters for 600 tokens
    
    if len(prompt) <= max_chars:
        return prompt
    
    # Trim to max_chars and try to end at a sentence boundary
    trimmed = prompt[:max_chars]
    
    # Try to find the last sentence ending
    last_period = trimmed.rfind('.')
    last_question = trimmed.rfind('?')
    last_exclaim = trimmed.rfind('!')
    
    # Get the furthest sentence ending
    last_sentence_end = max(last_period, last_question, last_exclaim)
    
    if last_sentence_end > max_chars * 0.8:  # If we can keep at least 80% of content
        trimmed = trimmed[:last_sentence_end + 1]
    
    return trimmed


def trim_config_prompts(input_file: str, output_file: str, max_tokens: int = 600):
    """
    Read a config JSON, trim prompts that exceed max_tokens, and save to output file.
    """
    # Load the config
    with open(input_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Trim prompts
    original_count = len(config['prompts'])
    trimmed_count = 0
    
    for i, prompt in enumerate(config['prompts']):
        original_length = len(prompt)
        trimmed_prompt = trim_prompt(prompt, max_tokens)
        
        if len(trimmed_prompt) < original_length:
            config['prompts'][i] = trimmed_prompt
            trimmed_count += 1
            print(f"Prompt {i+1}: Trimmed from {original_length} chars to {len(trimmed_prompt)} chars")
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessed {original_count} prompts")
    print(f"Trimmed {trimmed_count} prompts")
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Trim prompts in config JSON to specified token limit"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input config JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output config JSON file (default: overwrites input)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum tokens per prompt (default: 600)"
    )
    
    args = parser.parse_args()
    
    output_file = args.output if args.output else args.input
    
    trim_config_prompts(args.input, output_file, args.max_tokens)


if __name__ == "__main__":
    main()
