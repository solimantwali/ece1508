import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from PolicyModel import PolicyValueModel, compute_seq_logprobs


def load_dataset_with_rewards(json_path, tokenizer_name, max_length=256):
    """
    JSON format:
    [
      { "sentence": "...", "model_response": "...", "reward": 0.7 },
      ...
    ]

    For this script, we only care about tokenizing the model responses,
    because ref_logprobs are computed over those.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    responses = [d["model_response"] for d in data]
    rewards = torch.tensor([d["reward"] for d in data], dtype=torch.float32)

    enc = tokenizer(
        responses,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    return input_ids, attention_mask, rewards


def precompute_reference_logprobs(ref_policy, dataloader, device):
    """
    Precompute reference logprobs once (on GPU if available) in the same order
    as the dataloader.

    Returns:
        A Python list of floats, one per sample, in dataloader order.
    """
    print("Precomputing reference logprobs...")
    all_ref_logprobs = []

    ref_policy.eval()
    with torch.no_grad():
        for batch_idx, (batch_ids, batch_mask, _) in enumerate(dataloader):
            # move batch to device (GPU if available)
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)

            logits, _ = ref_policy(batch_ids, batch_mask)
            # assuming compute_seq_logprobs returns shape [batch_size]
            ref_logprobs = compute_seq_logprobs(logits, batch_ids)

            # move to CPU, extend as Python floats
            all_ref_logprobs.extend(ref_logprobs.cpu().tolist())

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    print(f"Finished precomputing reference logprobs for {len(all_ref_logprobs)} samples.")
    return all_ref_logprobs


def main():
    # Hardcoded config (change paths/names as needed)
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "SFT_Model"
    json_path = "oem_train_baseline_first_1k.json"
    output_ref_logprobs_path = "oem_train_ref_logprobs.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (responses + rewards, though rewards are unused here)
    print("Loading dataset...")
    input_ids, attention_mask, rewards = load_dataset_with_rewards(
        json_path,
        tokenizer_name=base_model_name,
    )

    # You can increase this if you have plenty of VRAM
    batch_size = 4
    dataset = TensorDataset(input_ids, attention_mask, rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Load reference (SFT) model on GPU (or CPU if no CUDA)
    print("Loading reference (SFT) policy...")
    ref_policy = PolicyValueModel(base_model_name, adapter_path).to(device)
    print("Ref policy device:", next(ref_policy.parameters()).device)

    # Freeze it
    for p in ref_policy.parameters():
        p.requires_grad = False

    # Optional: set eval mode and disable grad globally for safety
    ref_policy.eval()
    torch.set_grad_enabled(False)

    # Precompute reference logprobs in dataloader order
    ref_logprobs_list = precompute_reference_logprobs(ref_policy, dataloader, device)

    # Save to JSON
    out_path = Path(output_ref_logprobs_path)
    print(f"Saving reference logprobs to {out_path} ...")
    out_path.write_text(
        json.dumps(ref_logprobs_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Saved {len(ref_logprobs_list)} ref_logprobs entries to {out_path}.")


if __name__ == "__main__":
    main()
