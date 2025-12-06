import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ---------- CONFIG ----------
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

TRAIN_CSV = "oem_train.csv"
VAL_CSV = "oem_val.csv"

# OUTPUT_DIR = "mistral7b_oem_sft"
OUTPUT_DIR = "llama3_8b_oem_sft"

MAX_SEQ_LEN = 512  # if you ever get OOM, drop this to 256

MAX_TRAIN_EXAMPLES = 2874
MAX_VAL_EXAMPLES   = 152

SYSTEM_PROMPT = (
    "You are an honest, careful advice assistant. "
    "You prioritize long-term wellbeing over telling people what they want to hear. "
    "You give nuanced, non-sycophantic advice."
    "Respond in 5-6 sentences only, be concise and do not ramble on."
)
# ---------------------------


def load_oem_datasets(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    for col in ["sentence", "Human"]:
        if col not in train_df.columns or col not in val_df.columns:
            raise ValueError(f"Missing column '{col}' in train/val CSVs")

    if MAX_TRAIN_EXAMPLES is not None:
        train_df = train_df.sample(
            n=min(MAX_TRAIN_EXAMPLES, len(train_df)),
            random_state=42,
        ).reset_index(drop=True)

    if MAX_VAL_EXAMPLES is not None:
        val_df = val_df.sample(
            n=min(MAX_VAL_EXAMPLES, len(val_df)),
            random_state=42,
        ).reset_index(drop=True)

    print(f"Using {len(train_df)} train examples, {len(val_df)} val examples")

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    return train_ds, val_ds


def pick_device():
    if torch.cuda.is_available():
        print("Using device: cuda")
        return "cuda"
    if torch.backends.mps.is_available():
        print("Using device: mps")
        return "mps"
    print("Using device: cpu")
    return "cpu"


def main():
    device = pick_device()

    if device in ("cuda", "mps"):
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    # ----- tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- base model -----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=model_dtype,
    ).to(device)

    # ----- LoRA -----
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.gradient_checkpointing_enable()
    # model.config.use_cache = False

    # ----- data -----
    train_ds, val_ds = load_oem_datasets(TRAIN_CSV, VAL_CSV)

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

    train_ds = train_ds.map(format_example)
    val_ds = val_ds.map(format_example)

    def tokenize_example(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = train_ds.map(
        tokenize_example,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        tokenize_example,
        batched=True,
        remove_columns=val_ds.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    use_fp16 = False  # disable AMP for stability

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        save_total_limit=2,
        bf16=False,
        fp16=use_fp16,  # now always False
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training complete. LoRA adapters saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
