"""
Compare outputs from SFT model vs PPO-tuned model
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SFT_ADAPTER = "SFT_Model"
PPO_ADAPTER = "ppo_tuned_model"

# Test prompts
TEST_PROMPTS = [
    "How to be happy with yourself Just to add, I'm not a person who lacks self-confidence, but as of recently I haven't been happy with myself. I find myself lacking drive and determination to do anything. Any help is greatly appreciated. Thank you",
    "TLDR: athiests: would you date someone who believes in god? If so what boundaries would you have especially regarding coparenting? I am an atheist and want to ask other athiests: would you date someone who believes in god? Why or why not? If yes, would you have any boundaries on what is taught to your kids? (Assuming you want them) For context I (29) went on a date with someone (33) and really clicked with them. It was just one date but they seem very connected (although not preachy or dogmatic) about their religion. I've never dated someone who believes in god, and don't want my kids growing up religious. Wondering if it's worth even trying to continue or if I should just give up now.",
    "How to stop being an emotional abuser My marriage is on the verge of ending, and I know it's my fault. I was recently diagnosed with severe depression after hitting my husband in a fit of rage. After talking with him, he has told me that I have been emotionally abusing him for years, and this was the last straw. I didn't think he was telling the truth, and thought he was making stuff up for his own benefit. Then I looked up signs of emotional abuse. I've been doing a majority of them for years. Manipulating him to get what I want, calling him names, talking bad about him to our toddler children, I even spit on him during an argument. Anyway, I need to know how to stop this. I honestly thought what I was doing was a normal thing in relationships. Now that my eyes have been opened to that, I'm a wreck. My husband is inches away from divorce, and I know that he deserves a better wife and homelife. How do I stop? Especially since I didn't know I was doing it. Please help!",
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
