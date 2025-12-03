import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm

# === CONFIG ===
# BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# ADAPTER_DIR = "mistral7b_oem_sft"   # where your LoRA was saved

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "llama3_8b_oem_sft"      # where your LoRA was saved

# OEM_CSV = "oem_val.csv"             # or oem_train.csv, etc.
OEM_CSV = "oem_train.csv"             # or oem_train.csv, etc.
N_EXAMPLES = 3000

SYSTEM_PROMPT = (
    "You are an honest, careful advice assistant. "
    "You prioritize long-term wellbeing over telling people what they want to hear. "
    "You give nuanced, non-sycophantic advice."
    "You respond conversationally, like a human, and you are concise in your responses."
    # "Use at most 5-6 sentences in your response."
)

MAX_NEW_TOKENS = None
TEMPERATURE = 0.7
TOP_P = 0.9

OUTPUT_JSON = "oem_train_sft.json"
# OUTPUT_JSON = "oem_train_baseline.json"
# ===============


def pick_device():
    if torch.cuda.is_available():
        print("Using device: cuda")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using device: mps")
        return "mps"
    else:
        print("Using device: cpu")
        return "cpu"


def load_models(device):
    # One tokenizer for both
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    # Base model
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     BASE_MODEL,
    #     torch_dtype=dtype,
    # ).to(device)
    # base_model.eval()

    base_model=None

    # SFT model: base + LoRA adapters
    sft_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    ).to(device)
    sft_model = PeftModel.from_pretrained(sft_base, ADAPTER_DIR)
    sft_model.eval()

    # sft_model = None

    return tokenizer, base_model, sft_model


def build_prompt(tokenizer, sentence):
    """Builds the chat-style input text (no assistant message yet)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # tells the model: now it's the assistant's turn
    )
    return text


@torch.no_grad()
def generate_reply(model, tokenizer, device, sentence):
    prompt_text = build_prompt(tokenizer, sentence)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=TOP_P,
        temperature=TEMPERATURE,
    )

    # full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    # Simple heuristic: return everything after the user message
    # (for quick debugging it's fine if it's a bit messy)
    return reply


def main():
    device = pick_device()
    tokenizer, base_model, sft_model = load_models(device)

    # Load a few OEM examples
    df = pd.read_csv(OEM_CSV)
    # make sure we have both sentence + human
    for col in ["sentence", "Human"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {OEM_CSV}")

    # pick N_EXAMPLES random rows
    # examples = df.sample(n=min(N_EXAMPLES, len(df)), random_state=123).reset_index(drop=True)
    examples = df.iloc[:N_EXAMPLES].reset_index(drop=True)

    results = []
    with torch.inference_mode():
    # your whole generation loop
        for idx, row in tqdm(examples.iterrows(), total=len(examples)):
            sentence = row["sentence"]
            human = row["Human"]
    
            # print("=" * 80)
            # print(f"Example {idx + 1}")
            # print("-" * 80)
            # print("PROMPT (OEM sentence):")
            # print(sentence)
            # print("\nHUMAN ANSWER (ground truth):")
            # print(human)
    
            # print("\n[BASE MODEL ANSWER]")
            # base_reply = generate_reply(base_model, tokenizer, device, sentence)
            # print(base_reply)
    
            # print("\n[SFT MODEL ANSWER]")
            sft_reply = generate_reply(sft_model, tokenizer, device, sentence)
            # print(sft_reply)
            # print()
            results.append(
                {
                    "id": int(idx),
                    "sentence": sentence,
                    "human": human,
                    # "base_reply": base_reply,
                    "sft_reply": sft_reply,
                }
            )

    print("=" * 80)
    print("Done.")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
