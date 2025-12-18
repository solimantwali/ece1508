#
import json
from pathlib import Path
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import AutoTokenizer
from peft import LoraConfig

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
            snap[name] = p.detach().clone()
    return snap

def total_lora_l2_change(snap_before, model_after):
    total_sq = 0.0
    for name, p in model_after.named_parameters():
        if "lora_" not in name:
            continue
        if name not in snap_before:
            continue
        delta = (p.detach() - snap_before[name])
        total_sq += (delta.pow(2).sum()).item()
    return total_sq ** 0.5

class ResponseDataset(Dataset):
    """
    Holds raw model responses and scalar rewards.
    We tokenize in the collate_fn so we can do dynamic padding.
    """
    def __init__(self, responses, rewards):
        assert len(responses) == len(rewards)
        self.responses = responses
        # keep rewards as float list or tensor on CPU; move in collate
        self.rewards = rewards

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        return {
            "text": self.responses[idx],
            "reward": float(self.rewards[idx])
        }


def make_collate_fn(tokenizer, device, max_length=256):
    """
    Collate function that:
      - tokenizes with dynamic padding up to longest in batch
      - truncates at max_length
      - moves everything to GPU
    """
    def collate(batch):
        texts = [b["text"] for b in batch]
        rewards = torch.tensor(
            [b["reward"] for b in batch],
            dtype=torch.bfloat16
        )

        enc = tokenizer(
            texts,
            padding=True,          # pad to longest in this batch
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)
        rewards = rewards.to(device, non_blocking=True)

        return input_ids, attention_mask, rewards

    return collate


# -------------------------
# JSON loading (raw)
# -------------------------

def load_raw_dataset(json_path):
    """
    JSON format:
    [
      { "sentence": "...", "model_response": "...", "reward": 0.7 },
      ...
    ]
    Returns (responses: List[str], rewards: torch.Tensor[bfloat16] on CPU).
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    responses = [d["model_response"] for d in data]
    rewards = torch.tensor([d["reward"] for d in data], dtype=torch.bfloat16)
    return responses, rewards


# -------------------------
# Precompute logprobs helpers
# -------------------------

@torch.no_grad()
def precompute_reference_logprobs(ref_policy, dataloader, device):
    """
    Precompute average per-token logprobs under the (frozen) ref_policy.
    Shape: [N]
    """
    print("Precomputing reference logprobs...")
    ref_policy.eval()
    all_ref = []

    for batch_ids, batch_mask, _ in tqdm.tqdm(dataloader, desc="Ref logprobs", leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            ref_logits, _ = ref_policy(batch_ids, batch_mask)
        ref_logprobs = compute_seq_logprobs(ref_logits, batch_ids, average=True)  # [B]
        all_ref.append(ref_logprobs)

    ref_logprobs_all = torch.cat(all_ref, dim=0).to(device)
    print(f"Precomputed ref_logprobs shape: {ref_logprobs_all.shape}")
    return ref_logprobs_all


@torch.no_grad()
def precompute_policy_logprobs(policy, dataloader, device):
    """
    Precompute average per-token logprobs of the *current* policy on the whole dataset.
    Shape: [N]
    """
    print("Precomputing old policy logprobs...")
    policy.eval()
    all_logprobs = []

    for batch_ids, batch_mask, _ in tqdm.tqdm(dataloader, desc="Old logprobs", leave=False):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = policy(batch_ids, batch_mask)
        logprobs = compute_seq_logprobs(logits, batch_ids, average=True)  # [B]
        all_logprobs.append(logprobs.detach())

    policy.train()
    old_logprobs_all = torch.cat(all_logprobs, dim=0).to(device)
    print(f"Precomputed old_logprobs shape: {old_logprobs_all.shape}")
    return old_logprobs_all


def load_or_precompute_old_logprobs(policy, dataloader, device, path):
    """
    If a cache file exists, load old_logprobs from disk.
    Otherwise, compute them for the current policy and save.

    Treats the current policy as the fixed "behavior policy" for PPO.
    """
    if os.path.exists(path):
        print(f"Loading old_logprobs from {path} ...")
        old_logprobs_all = torch.load(path, map_location=device)
        print(f"Loaded old_logprobs: shape={old_logprobs_all.shape}")
    else:
        old_logprobs_all = precompute_policy_logprobs(policy, dataloader, device)
        torch.save(old_logprobs_all.cpu(), path)
        print(f"Saved old_logprobs to {path}")
    return old_logprobs_all.to(device)

def main():
    # params
    clip_range = 0.2
    value_coef = 0.5
    entropy_coef = 0.005
    # kl_coef = 0.005   # set to >0 if you want KL to reference
    kl_coef = 1e-4
    batch_size = 4
    num_epochs = 2
    lr = 1e-5
    max_length = 256

    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    ref_logprobs_path = "ref_logprobs.pt"
    old_logprobs_path = "old_logprobs.pt"

    base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    adapter_path = 'llama3_8b_oem_sft'
    json_path = "oem_train_with_rewards.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Load raw data & tokenizer
    # -------------------------
    responses, rewards = load_raw_dataset(json_path)
    N = len(responses)
    print(f"Loaded dataset: N={N}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset & DataLoader (dynamic padding in collate)
    dataset = ResponseDataset(responses, rewards)
    collate_fn = make_collate_fn(tokenizer, device, max_length=max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    print("Loading policy model...")
    policy = PolicyValueModel(
        base_model_name, 
        adapter_path, 
        dtype=torch.bfloat16,
        lora_config=lora_config
    ).to(device=device, dtype=torch.bfloat16)
    # policy = torch.compile(policy) 
    print("Policy dtype:", next(policy.base.parameters()).dtype)
    print("Policy device:", next(policy.base.parameters()).device)

    print("Before setting require grad")
    debug_lora_requires_grad(policy)

    # USE_LORA = adapter_path is not None 
    for name, p in policy.named_parameters():
        if "lora_" in name or "value_head" in name:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    print("After setting require grad")
    debug_lora_requires_grad(policy)
    debug_trainable_params(policy)


    # Enable gradient checkpointing to save memory
    # policy.base.gradient_checkpointing_enable()
    # policy.base.config.use_cache = False
    
    # -------------------------
    # Reference model + KL anchor (on CPU)
    # -------------------------
    print("Loading reference model on CPU...")
    ref_policy = PolicyValueModel(base_model_name, 
                                  adapter_path, 
                                  dtype=torch.bfloat16
                                  ).to(device=device, dtype=torch.bfloat16)
    print("Ref policy device:", next(ref_policy.parameters()).device)
    print("Ref policy dtype:", next(policy.parameters()).dtype)

    # Copy initial policy weights into reference model
    ref_policy.load_state_dict(policy.state_dict(), strict=False)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # Precompute reference logprobs once
    if os.path.exists(ref_logprobs_path):
        print(f"Loading precomputed ref_logprobs from {ref_logprobs_path} ...")
        ref_logprobs_all = torch.load(ref_logprobs_path, map_location=device)
        print(f"Loaded ref_logprobs: shape={ref_logprobs_all.shape}")
    else:
        print("No precomputed ref_logprobs found. Computing now...")
        ref_logprobs_all = precompute_reference_logprobs(ref_policy, dataloader, device)
        torch.save(ref_logprobs_all.cpu(), ref_logprobs_path)
        print(f"Saved reference logprobs to {ref_logprobs_path}")

    # Move to device for training
    # We don't need the ref policy anymore during training
    del ref_policy
    torch.cuda.empty_cache()

    # -------------------------
    # Precompute OLD logprobs once (behavior policy)
    # -------------------------
    old_logprobs_all = load_or_precompute_old_logprobs(
        policy, dataloader, device, old_logprobs_path
    )

    # -------------------------
    # Optimizer & PPO hyperparams
    # -------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=lr,
    )

    print("Taking initial LoRA snapshot...")
    lora_before = snapshot_lora_params(policy)

    # -------------------------
    # PPO training loop
    # -------------------------
    for epoch in range(num_epochs):
        epoch_policy_losses = []
        epoch_value_losses = []

        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        cursor = 0

        for batch_ids, batch_mask, batch_rewards in pbar:
            bsz = batch_ids.size(0)
            ref_logprobs = ref_logprobs_all[cursor:cursor + bsz]
            old_logprobs = old_logprobs_all[cursor:cursor + bsz]
            cursor += bsz

            # Single forward pass with grad
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_new, values_new = policy(batch_ids, batch_mask)

            values_new = values_new.float()  # [B]
            new_logprobs = compute_seq_logprobs(
                logits_new, batch_ids, average=True
            )  # [B]

            # Baseline: use current value head, detached
            values_old = values_new.detach()

            # Returns = rewards (no bootstrap here)
            returns = batch_rewards.float()

            # Advantages
            advantages = returns - values_old
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = torch.clamp(advantages, -10, 10)

            # PPO ratio and clipped objective
            ratio = torch.exp(new_logprobs - old_logprobs)  # [B]
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            # Value loss
            value_loss = F.mse_loss(values_new, returns)

            # Entropy bonus
            logits_fp32 = logits_new.float()
            log_probs = torch.log_softmax(logits_fp32, dim=-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            # KL to reference (in logprob space, per-sample)
            kl = torch.tensor(0.0, device=device)
            if kl_coef > 0.0:
                kl = (new_logprobs - ref_logprobs).mean()

            # Total loss
            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + kl_coef * kl
            )

            optimizer.zero_grad()

            if torch.isnan(loss):
                print(f"WARNING: NaN loss at epoch {epoch}, cursor {cursor}")
                print(f"  policy_loss={policy_loss.item()}, value_loss={value_loss.item()}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_policy_losses.append(policy_loss.item())
            epoch_value_losses.append(value_loss.item())

            pbar.set_postfix({
                "pol": f"{policy_loss.item():.3f}",
                "val": f"{value_loss.item():.3f}",
                "ent": f"{entropy.item():.3f}",
                "kl":  f"{kl.item():.3f}",
            })

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
