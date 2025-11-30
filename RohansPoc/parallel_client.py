import asyncio
import httpx

SERVERS = [
    ("gpt2-A", "http://localhost:8001/v1/chat/completions"),
    ("gpt2-B", "http://localhost:8002/v1/chat/completions"),
]

PROMPTS = [
    "Explain PPO in one short paragraph.",
    "Give a 2-bullet explanation of policy gradients.",
    "What is the role of the KL penalty in RLHF PPO?",
    "Summarize Monte Carlo policy evaluation in 3 sentences.",
]

async def query_one(server_name: str, base_url: str, prompt: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": "gpt2",  # vLLM maps this to the loaded model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 64,
            "temperature": 0.7,
        }
        r = await client.post(base_url, json=payload)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return server_name, prompt, text

async def main():
    tasks = []
    # Round-robin prompts across the two servers
    for i, prompt in enumerate(PROMPTS):
        name, url = SERVERS[i % len(SERVERS)]
        tasks.append(query_one(name, url, prompt))

    for coro in asyncio.as_completed(tasks):
        name, prompt, text = await coro
        print(f"\n=== {name} ===")
        print(f"Prompt: {prompt}")
        print(f"Answer: {text}\n")

if __name__ == "__main__":
    asyncio.run(main())
