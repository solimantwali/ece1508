import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------- CONFIG ----------
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
VAL_CSV = "oem_val.csv"

SYSTEM_PROMPT = (
    "You are an honest, careful advice assistant. "
    "You prioritize long-term wellbeing over telling people what they want to hear. "
    "You give nuanced, non-sycophantic advice. "
    "Be concise and do not ramble on."
)

MAX_INPUT_LEN = 256        # max tokens for prompt
MAX_NEW_TOKENS = 256       # max generated tokens
BATCH_SIZE = 4
# ---------------------------


def pick_device_and_dtype():
    if torch.cuda.is_available():
        print("Using device: cuda")
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        print("Using device: mps")
        return "mps", torch.float16
    print("Using device: cpu")
    return "cpu", torch.float32


def load_val_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sentence" not in df.columns:
        raise ValueError("VAL_CSV must contain a 'sentence' column.")
    # Human column is optional but useful to pass through to JSON
    return df


def load_model_and_tokenizer(variant: str, device: str, dtype: torch.dtype):
    """
    variant in {"base", "sft", "sft_ppo", "baseline_ppo"}.
    """
    adapter_map = {
        "base": None,
        "sft": "llama3_8b_oem_sft",
        "sft_ppo": "ppo_tuned_model",
        "baseline_ppo": "ppo_baseline_tuned_model",
    }
    if variant not in adapter_map:
        raise ValueError(f"Unknown variant '{variant}'. Valid: {list(adapter_map.keys())}")

    adapter_dir = adapter_map[variant]
    print(f"Loading base model: {BASE_MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
    )

    if adapter_dir is not None:
        print(f"Loading LoRA adapter from: {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = base_model

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return model, tokenizer


def build_prompts(df: pd.DataFrame, tokenizer: AutoTokenizer):
    """
    Build chat prompts (system + user) for each row in the val dataframe.
    Returns a list of strings (already chat-templated).
    """
    prompts = []
    for _, row in df.iterrows():
        user_text = str(row["sentence"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # ends with assistant header
        )
        prompts.append(prompt_text)
    return prompts

def clean_response(text: str) -> str:
    """Remove echoed chat template junk the model sometimes regenerates."""
    # Drop everything before the *last* assistant tag, if present
    for marker in ["assistant\n\n", "assistant\n", "assistant:"]:
        idx = text.lower().rfind(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            break

    # Remove echoed SYSTEM or USER blocks if they slipped through
    text = text.replace("system\n", "")
    text = text.replace("user\n", "")
    text = text.replace("assistant\n", "")

    return text.strip()

@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts,
    device: str,
    # max_input_len: int = MAX_INPUT_LEN,
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
):
    """
    prompts: list of prompt strings (chat-templated).
    Returns list of decoded assistant responses (strings).
    """
    all_outputs = []
    # for start in range(0, len(prompts), batch_size):
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[start : start + batch_size]

        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=None,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        input_lengths = attention_mask.sum(dim=1)  # [B]

        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            # do_sample=False,          # deterministic / greedy for eval
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.15,     
            # temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        for i in range(gen_out.size(0)):
            seq = gen_out[i]
            inp_len = input_lengths[i].item()
            gen_tokens = seq[inp_len:]  # only the newly generated tokens
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            # all_outputs.append(text)
            clean = clean_response(text)
            all_outputs.append(clean)

    return all_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        type=str,
        default="base",
        choices=["base", "sft", "sft_ppo", "baseline_ppo"],
        help="Which model variant to use.",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=VAL_CSV,
        help="Path to oem_val.csv",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="oem_val_generations.json",
        help="Where to write the JSON output.",
    )
    args = parser.parse_args()

    device, dtype = pick_device_and_dtype()

    df_val = load_val_df(args.val_csv)
    model, tokenizer = load_model_and_tokenizer(args.variant, device, dtype)

    print(f"Building prompts from {len(df_val)} validation rows...")
    prompts = build_prompts(df_val, tokenizer)

    print("Generating responses...")
    model_responses = generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
    )

    assert len(model_responses) == len(df_val)

    # Build JSON objects
    results = []
    for idx, ( (_, row), resp ) in enumerate(zip(df_val.iterrows(), model_responses)):
        item = {
            "id": int(idx),
            "sentence": str(row["sentence"]),
            # include Human answer if available, for later analysis
            "human_response": str(row["Human"]) if "Human" in row else None,
            "model_response": resp,
            "model_variant": args.variant,
        }
        results.append(item)

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} generations to {out_path}")


if __name__ == "__main__":
    main()