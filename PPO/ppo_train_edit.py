#
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


def precompute_reference_logprobs(ref_policy, dataloader):
    """
    Precompute reference logprobs ONCE using the (frozen) ref_policy.
    We keep everything on CPU to avoid wasting GPU memory.
    """
    print("Precomputing reference logprobs...")
    all_ref_logprobs = []

    ref_policy.eval()
    with torch.no_grad():
        for batch_ids, batch_mask, _ in dataloader:
            # ensure on CPU
            batch_ids_cpu = batch_ids.cpu()
            batch_mask_cpu = batch_mask.cpu()

            ref_logits, _ = ref_policy(batch_ids_cpu, batch_mask_cpu)
            # compute_seq_logprobs should return [batch_size] or [batch_size, ...]
            ref_logprobs = compute_seq_logprobs(ref_logits, batch_ids_cpu)
            all_ref_logprobs.append(ref_logprobs)

    ref_logprobs_all = torch.cat(all_ref_logprobs, dim=0).cpu()
    print(f"Precomputed ref_logprobs shape: {ref_logprobs_all.shape}")
    return ref_logprobs_all


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


def load_dataset_with_rewards(json_path, tokenizer_name, max_length=16):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SFT model configuration
    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = "SFT_Model"
    json_path = "oem_val_sft_with_rewards.json"

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
    
    # reference (KL anchor) = copy of initial policy (keep on CPU to save GPU memory)
        # -------------------------
    # Reference model + KL anchor
    # -------------------------
    print("Loading reference model on CPU...")
    ref_policy = PolicyValueModel(base_model_name, adapter_path)
    print("Ref policy device:", next(ref_policy.parameters()).device)

    # Copy initial policy weights into reference model
    ref_policy.load_state_dict(policy.state_dict(), strict=False)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False
    ref_logprobs_path = "ref_logprobs.pt"
    # Precompute reference logprobs once
    if os.path.exists(ref_logprobs_path):
        print(f"Loading precomputed reference logprobs from {ref_logprobs_path} ...")
        ref_logprobs_all = torch.load(ref_logprobs_path, map_location="cpu")
        print(f"Loaded ref_logprobs: shape={ref_logprobs_all.shape}")
    else:
        print("No precomputed ref_logprobs found. Computing now...")
        ref_logprobs_all = precompute_reference_logprobs(ref_policy, dataloader)
        torch.save(ref_logprobs_all, ref_logprobs_path)
        print(f"Saved reference logprobs to {ref_logprobs_path}")

    # Move to device for training
    ref_logprobs_all = ref_logprobs_all.to(device)

    # We don't need the ref policy anymore during training
    del ref_policy
    torch.cuda.empty_cache()

    # -------------------------
    # Optimizer & PPO hyperparams
    # -------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=1e-5,
    )

    clip_range = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    kl_coef = 0.05   # set to >0 if you want KL to reference

    num_epochs = 1
    batch_size = dataloader.batch_size  # used to slice ref_logprobs_all

    print("Taking initial LoRA snapshot...")
    lora_before = snapshot_lora_params(policy)

    # -------------------------
    # PPO training loop
    # -------------------------
    for epoch in range(num_epochs):
        # track per-epoch losses
        epoch_policy_losses = []
        epoch_value_losses = []

        # Make a frozen copy of the current policy = "old_policy"
        old_policy = PolicyValueModel(base_model_name, adapter_path).to(device)
        old_policy.load_state_dict(policy.state_dict())
        old_policy.eval()
        for p in old_policy.parameters():
            p.requires_grad_(False)

        for batch_idx, (batch_ids, batch_mask, batch_rewards) in enumerate(dataloader):
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_rewards = batch_rewards.to(device)

            # -------------------------
            # Old policy logprobs & values
            # -------------------------
            with torch.no_grad():
                logits_old, values_old = old_policy(batch_ids, batch_mask)
                old_logprobs = compute_seq_logprobs(logits_old, batch_ids)  # shape [B] or [B, ...]

            # Returns = rewards (no bootstrap here)
            returns = batch_rewards

            # Advantages
            advantages = returns - values_old
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -10, 10)

            # -------------------------
            # New policy logprobs & values
            # -------------------------
            logits_new, values_new = policy(batch_ids, batch_mask)
            new_logprobs = compute_seq_logprobs(logits_new, batch_ids)

            # PPO ratio and clipped objective
            ratio = torch.exp(new_logprobs - old_logprobs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            # Value loss
            value_loss = F.mse_loss(values_new, returns)

            # Entropy bonus (encourage diversity)
            probs = torch.softmax(logits_new, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()

            # -------------------------
            # KL to precomputed reference logprobs
            # -------------------------
            # Slice ref_logprobs_all to get the entries for this batch
            start = batch_idx * batch_size
            end = start + batch_ids.size(0)  # handle final batch < batch_size
            ref_logprobs = ref_logprobs_all[start:end]

            kl = torch.tensor(0.0, device=device)
            if kl_coef > 0.0:
                # old_logprobs and ref_logprobs must have same shape and device
                kl = (old_logprobs - ref_logprobs).mean()

            # -------------------------
            # Total loss
            # -------------------------
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
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

        # -------------------------
        # Epoch summary
        # -------------------------
        avg_policy_loss = (
            sum(epoch_policy_losses) / len(epoch_policy_losses)
            if epoch_policy_losses else float("nan")
        )
        avg_value_loss = (
            sum(epoch_value_losses) / len(epoch_value_losses)
            if epoch_value_losses else float("nan")
        )

        print(
            f"Epoch {epoch}: "
            f"policy_loss={avg_policy_loss:.4f}, "
            f"value_loss={avg_value_loss:.4f}, "
            f"entropy={entropy.item():.4f}, "
            f"kl={kl.item():.4f}"
        )
        # ---- CLEANUP: free the old_policy model before next epoch ----
        del old_policy
        torch.cuda.empty_cache()

    # -------------------------
    # After training
    # -------------------------
    print("Computing in-run LoRA change...")
    lora_change_l2 = total_lora_l2_change(lora_before, policy)
    print(f"[DEBUG] Total LoRA L2 change during PPO run: {lora_change_l2:.6e}")

    # Save PPO-updated model
    print("Saving PPO-tuned model...")
    policy.base.save_pretrained("ppo_tuned_model")  # Saves LoRA adapters
    torch.save(policy.value_head.state_dict(), "ppo_tuned_model/value_head.pt")
    print("Model saved to ppo_tuned_model/")



if __name__ == "__main__":
    main()
