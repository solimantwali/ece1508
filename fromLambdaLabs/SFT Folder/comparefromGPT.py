import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm
import pandas as pd

# === CONFIG ===
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "llama3_8b_oem_sft"
OEM_CSV = "oem_val.csv"
N_EXAMPLES = 5

SYSTEM_PROMPT = (
    "You are an honest, careful advice assistant. "
    "You prioritize long-term wellbeing over telling people what they want to hear. "
    "You give nuanced, non-sycophantic advice."
    "You respond conversationally, like a human, and you are concise in your responses."
)

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
BATCH_SIZE = 8  # Process multiple examples at once

OUTPUT_JSON = "oem_val_baseline_optimized.json"
# ===============


def load_model(device="cuda"):
    """Load model with optimizations"""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use device_map for automatic memory optimization
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,  # Enable KV-cache
    )
    model = base_model
    # model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    
    return tokenizer, model


def build_prompts(tokenizer, sentences):
    """Build prompts for a batch of sentences"""
    all_messages = []
    for sentence in sentences:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ]
        all_messages.append(messages)
    
    # Batch tokenize all prompts at once
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]
    return prompts


@torch.no_grad()
def generate_batch(model, tokenizer, prompts):
    """Generate responses for a batch of prompts"""
    # Tokenize batch with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS*BATCH_SIZE,
        do_sample=True,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode only the generated part (skip input)
    input_lengths = inputs["input_ids"].shape[1]
    responses = []
    for output in outputs:
        gen_tokens = output[input_lengths:]
        response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        responses.append(response)
    
    return responses


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer, model = load_model(device)
    
    # Load data more efficiently
    df = pd.read_csv(OEM_CSV, nrows=N_EXAMPLES)
    
    # Extract columns as lists (more efficient than iterrows)
    sentences = df["sentence"].tolist()
    humans = df["Human"].tolist()
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(sentences), BATCH_SIZE), desc="Generating responses"):
        batch_sentences = sentences[i:i+BATCH_SIZE]
        batch_humans = humans[i:i+BATCH_SIZE]
        
        # Generate batch
        prompts = build_prompts(tokenizer, batch_sentences)
        responses = generate_batch(model, tokenizer, prompts)
        
        # Store results
        for j, (sentence, human, response) in enumerate(zip(batch_sentences, batch_humans, responses)):
            results.append({
                "id": i + j,
                "sentence": sentence,
                "human": human,
                "sft_reply": response,
            })
    
    print("Done. Saving results...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
