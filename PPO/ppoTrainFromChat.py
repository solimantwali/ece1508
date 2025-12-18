import json
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from PolicyModel import PolicyValueModel, compute_seq_logprobs


def debug_lora_grads(model):
    total_lora = 0
    nonzero_grad_lora = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            total_lora += 1
            if p.grad is not None and p.grad.abs().sum().item() > 0:
                nonzero_grad_lora += 1
    print(f"[DEBUG] LoRA params with nonzero grad: {nonzero_grad_lora}/{total_lora}")


def debug_lora_requires_grad(model):
    total, trainable = 0, 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            total += 1
            if p.requires_grad:
                trainable += 1
    print(f"[DEBUG] LoRA params with requires_grad=True: {trainable}/{total}")


def debug_trainable_params(model):
    total = 0
    trainable = 0
    lora_trainable = 0
    base_trainable = 0
    value_trainable = 0
    
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            # Count categories
            if "lora_" in name:
                lora_trainable += p.numel()
            elif "value_head" in name:
                value_trainable += p.numel()
            else:
                base_trainable += p.numel()

    print("\n===== Trainable Parameter Report =====")
    print(f"Total parameters:         {total:,}")
    print(f"Trainable parameters:     {trainable:,}")
    print(f"Frozen parameters:        {total - trainable:,}")
    print("--------------------------------------")
    print(f"LoRA trainable params:    {lora_trainable:,}")
    print(f"Value head trainable:     {value_trainable:,}")
    print(f"Base model trainable:     {base_trainable:,}")
    print("======================================\n")


def snapshot_lora_params(model):
    snap = {}
    for name, p in model.named_parameters():
        if "lora_" in name:
            if getattr(p, "is_meta", False):
                continue
            snap[name] = p.detach().cpu().clone()
    return snap


def total_lora_l2_change(snap_before, model_after):
    import math
    total_sq = 0.0
    for name, p in model_after.named_parameters():
        if "lora_" not in name:
            continue
        if name not in snap_before:
            continue
        delta = (p.detach().cpu() - snap_before[name])
        total_sq += float(delta.pow(2).sum())
    return math.sqrt(total_sq)


def load_dataset_with_rewards(json_path, tokenizer_name, max_length=256):
    """
    JSON format:
    [
      { "sentence": "...", "model_response": "...", "reward": 0.7 },
      ...
    ]
    
    For RLHF/PPO, we only tokenize the responses since those are what we're optimizing.
    The rewards are already computed based on these responses.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    responses = [d["model_response"] for d in data]
    rewards = torch.tensor([d["reward"] for d in data], dtype=torch.float32)

    # Tokenize only the responses - these are what we're training on
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


def load_ref_logprobs_from_json(ref_logprobs_path: str, expected_len: int) -> torch.Tensor:
    """
    Load precomputed reference logprobs from a JSON file and return as a 1D float32 tensor.
    Assumes the JSON is a list of floats, ordered the same way as the dataset.
    """
    path = Path(ref_logprobs_path)
    print(f"Loading reference logprobs from {path} ...")
    ref_list = json.loads(path.read_text(encoding="utf-8"))
    ref_tensor = torch.tensor(ref_list, dtype=torch.float32)

    if ref_tensor.shape[0] != expected_len:
        raise ValueError(
            f"ref_logprobs length mismatch: got {ref_tensor.shape[0]}, "
            f"expected {expected_len}"
        )

    print(f"Loaded {ref_tensor.shape[0]} reference logprobs entries.")
    return ref_tensor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SFT model configuration
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "SFT_Model"
    json_path = "oem_val_sft_with_rewards.json"
    ref_logprobs_json_path = "oem_val_ref_logprobs.json"  # precomputed JSON

    input_ids, attention_mask, rewards = load_dataset_with_rewards(
        json_path,
        tokenizer_name=base_model_name,
    )
    batch_size = 1
    dataset = TensorDataset(input_ids, attention_mask, rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size)  # Minimum 2 for std calculation

    print("Loading policy model...")
    policy = PolicyValueModel(base_model_name, adapter_path).to(device)
    print("Policy device:", next(policy.parameters()).device)

    print("Before setting require grad")
    debug_lora_requires_grad(policy)
    for name, p in policy.named_parameters():
        if "lora_" in name or "value_head" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    print("After setting require grad")
    debug_lora_requires_grad(policy)
    debug_trainable_params(policy)
    # Enable gradient checkpointing to save memory
    policy.base.gradient_checkpointing_enable()
    
    # Load precomputed reference logprobs instead of recomputing
    num_samples = len(dataset)
    ref_logprobs_all = load_ref_logprobs_from_json(ref_logprobs_json_path, num_samples)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=1e-5,
    )

    clip_range = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    kl_coef = 0.05   # set >0 if you want KL to reference

    num_epochs = 3

    print("Taking initial LoRA snapshot...")
    lora_before = snapshot_lora_params(policy)

    for epoch in range(num_epochs):
        # Make a frozen copy of the current policy = "old_policy"
        old_policy = PolicyValueModel(base_model_name, adapter_path).to(device)
        old_policy.load_state_dict(policy.state_dict())
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        # reset running losses if you track them per epoch
        epoch_policy_losses = []
        epoch_value_losses = []

        # track position in ref_logprobs_all across batches
        sample_offset = 0

        for batch_idx, (batch_ids, batch_mask, batch_rewards) in enumerate(dataloader):
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_rewards = batch_rewards.to(device)
            current_bs = batch_ids.size(0)

            with torch.no_grad():
                logits_old, values_old = old_policy(batch_ids, batch_mask)
                old_logprobs = compute_seq_logprobs(logits_old, batch_ids)

            returns = batch_rewards
            advantages = returns - values_old

            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -10, 10)

            logits_new, values_new = policy(batch_ids, batch_mask)
            new_logprobs = compute_seq_logprobs(logits_new, batch_ids)

            ratio = torch.exp(new_logprobs - old_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.mean(torch.min(unclipped, clipped))
            # value loss
            value_loss = F.mse_loss(values_new, returns)

            # entropy bonus (encourage diversity)
            probs = torch.softmax(logits_new, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()

            # KL to reference model using precomputed ref_logprobs_all
            if kl_coef > 0.0:
                # Take the slice of ref_logprobs corresponding to this batch
                batch_ref_logprobs = ref_logprobs_all[
                    sample_offset : sample_offset + current_bs
                ].to(device)
                kl = (old_logprobs - batch_ref_logprobs).mean()
            else:
                kl = torch.tensor(0.0, device=device)

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + kl_coef * kl
            )

            optimizer.zero_grad()
            
            # Check for NaN before backward
            if torch.isnan(loss):
                print(f"WARNING: NaN loss detected at epoch {epoch}, batch {batch_idx}")
                print(f"  policy_loss={policy_loss.item()}, value_loss={value_loss.item()}")
                print(f"  advantages: {advantages}")
                print(f"  ratio: {ratio}")
                sample_offset += current_bs
                continue
            
            loss.backward()
            # debug_lora_grads(policy)

            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

            # advance offset into ref_logprobs_all
            sample_offset += current_bs

        avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses) if epoch_policy_losses else float('nan')
        avg_value_loss = sum(epoch_value_losses) / len(epoch_value_losses) if epoch_value_losses else float('nan')
        
        print(
            f"Epoch {epoch}: "
            f"policy_loss={avg_policy_loss:.4f}, "
            f"value_loss={avg_value_loss:.4f}, "
            f"entropy={entropy.item():.4f}, "
            f"kl={kl.item():.4f}"
        )

        # free old_policy at end of epoch
        del old_policy
        torch.cuda.empty_cache()

    print("Computing in-run LoRA change...")
    lora_change_l2 = total_lora_l2_change(lora_before, policy)
    print(f"[DEBUG] Total LoRA L2 change during PPO run: {lora_change_l2:.6e}")

    # Save PPO-updated model
    print("Saving PPO-tuned model...")
    policy.base.save_pretrained("ppo_tuned_model")  # Saves LoRA adapters
    torch.save(policy.value_head.state_dict(), "ppo_tuned_model/value_head.pt")  # Save value head
    print("Model saved to ppo_tuned_model/")


if __name__ == "__main__":
    main()
