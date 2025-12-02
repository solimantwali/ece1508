import asyncio
import httpx
import json
import argparse
from pathlib import Path

async def query_one(server_name: str, base_url: str, prompt: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": "gpt2",  # vLLM maps this to the loaded model
            "messages": [
                {"role": "system", "content": "You are to give the user feedback and advice on their scenario."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 64,
            "temperature": 0,
        }
        r = await client.post(base_url, json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return server_name, prompt, text

async def process_server(server_name: str, base_url: str, output_file: str, prompts: list):
    """Process all prompts for a single server and save results to JSON."""
    results = []
    for prompt in prompts:
        name, prompt_text, answer = await query_one(server_name, base_url, prompt)
        result_obj = {
            "server": server_name,
            "model": "gpt2",
            "prompt": prompt_text,
            "response": answer,
        }
        results.append(result_obj)
        print(f"[{server_name}] Processed: {prompt_text[:50]}...")
    
    # Write results to JSON file
    output_path = Path(output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[{server_name}] Results saved to {output_file}")

def load_config(config_path: str):
    """Load servers and prompts from JSON config file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    servers = [
        (srv["name"], srv["url"], srv["output_file"]) 
        for srv in config["servers"]
    ]
    prompts = config["prompts"]
    
    return servers, prompts

async def main():
    parser = argparse.ArgumentParser(description="Run parallel inference across multiple vLLM servers")
    parser.add_argument("--config", type=str, default="InputData/smallTest/in_config.json", 
                       help="Path to JSON config file (default: config.json)")
    args = parser.parse_args()
    
    # Load configuration
    servers, prompts = load_config(args.config)
    
    # Shard prompts across servers
    num_servers = len(servers)
    sharded_prompts = [[] for _ in range(num_servers)]
    for i, prompt in enumerate(prompts):
        sharded_prompts[i % num_servers].append(prompt)
    
    tasks = []
    # Create one task per server with its shard of prompts
    for idx, (server_name, base_url, output_file) in enumerate(servers):
        tasks.append(process_server(server_name, base_url, output_file, sharded_prompts[idx]))
    
    # Run both servers in parallel
    await asyncio.gather(*tasks)
    print("\nAll servers finished processing.")

if __name__ == "__main__":
    asyncio.run(main())
