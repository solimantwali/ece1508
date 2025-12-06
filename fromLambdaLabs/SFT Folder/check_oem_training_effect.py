import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel

# ---- CONFIG ----
# BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.2"
# ADAPTER_DIR  = "mistral7b_oem_sft"     # your SFT output dir

BASE_MODEL   = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR  = "llama3_8b_oem_sft"      # your SFT output dir

VAL_CSV      = "oem_val.csv"           # or oem_train.csv if you want
MAX_SEQ_LEN  = 512
SYSTEM_PROMPT = (
    "You are an honest, careful advice assistant. "
    "You prioritize long-term wellbeing over telling people what they want to hear. "
    "You give nuanced, non-sycophantic advice."
)
# --------------


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


def load_val_dataset(csv_path, tokenizer):
    df = pd.read_csv(csv_path)
    for col in ["sentence", "Human"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")
    df = df[["sentence", "Human"]].dropna().reset_index(drop=True)
    ds = Dataset.from_pandas(df)

    # same formatting as training
    def format_example(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["sentence"]},
            {"role": "assistant", "content": example["Human"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(format_example)

    def tokenize_example(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(
        tokenize_example,
        batched=True,
        remove_columns=ds.column_names,
    )
    return ds


def make_trainer(model, tokenizer, eval_ds, device):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    training_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer


def main():
    device = pick_device()

    # tokenizer (shared)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # build eval dataset once
    eval_ds = load_val_dataset(VAL_CSV, tokenizer)

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    # ---- BASE MODEL ----
    print("\nLoading BASE model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    ).to(device)
    base_model.eval()

    base_trainer = make_trainer(base_model, tokenizer, eval_ds, device)
    base_metrics = base_trainer.evaluate()
    base_loss = base_metrics["eval_loss"]
    print(f"\nBASE model eval_loss: {base_loss:.4f}")

    # ---- SFT MODEL (base + LoRA) ----
    print("\nLoading SFT (LoRA) model...")
    sft_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
    ).to(device)
    sft_model = PeftModel.from_pretrained(sft_base, ADAPTER_DIR)
    sft_model.eval()

    sft_trainer = make_trainer(sft_model, tokenizer, eval_ds, device)
    sft_metrics = sft_trainer.evaluate()
    sft_loss = sft_metrics["eval_loss"]
    print(f"\nSFT model eval_loss:  {sft_loss:.4f}")

    # quick comparison
    diff = base_loss - sft_loss
    print("\n----------------------------")
    print(f"Loss improvement (base - SFT): {diff:.4f}")
    print("----------------------------")


if __name__ == "__main__":
    main()
