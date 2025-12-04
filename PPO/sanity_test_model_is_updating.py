"""
Compare outputs from SFT model vs PPO-tuned model
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SFT_ADAPTER = "importFiles"
PPO_ADAPTER = "ppo_tuned_model"

# Test prompts
TEST_PROMPTS = [
    "How to be happy with yourself",
    "What to do with a gf that doesn't know what she wants to do",
    "How do I stop being an emotional abuser",
]

def load_model(adapter_path):
    """Load base model with adapters"""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model

def generate_response(tokenizer, model, prompt):
    """Generate a response to the prompt"""
    messages = [
        {"role": "system", "content": "You are an honest, careful advice assistant. You prioritize long-term wellbeing over telling people what they want to hear. You give nuanced, non-sycophantic advice."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the assistant's response
    response = response.split("assistant")[-1].strip()
    return response

def main():
    print("Loading SFT model...")
    sft_tokenizer, sft_model = load_model(SFT_ADAPTER)
    
    print("Loading PPO model...")
    ppo_tokenizer, ppo_model = load_model(PPO_ADAPTER)
    
    results = []
    
    for prompt in TEST_PROMPTS:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}")
        
        print("\n--- SFT Response ---")
        sft_response = generate_response(sft_tokenizer, sft_model, prompt)
        print(sft_response)
        
        print("\n--- PPO Response ---")
        ppo_response = generate_response(ppo_tokenizer, ppo_model, prompt)
        print(ppo_response)
        
        results.append({
            "prompt": prompt,
            "sft_response": sft_response,
            "ppo_response": ppo_response
        })
    
    # Save results
    with open("sft_vs_ppo_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n\nResults saved to sft_vs_ppo_comparison.json")

if __name__ == "__main__":
    main()
