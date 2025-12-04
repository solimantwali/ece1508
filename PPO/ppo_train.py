#
import json
from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from PolicyModel import PolicyValueModel, compute_seq_logprobs


def load_dataset_with_rewards(json_path, tokenizer_name, max_length=512):
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
    adapter_path = "importFiles"
    json_path = "oem_val_sft_with_rewards.json"

    input_ids, attention_mask, rewards = load_dataset_with_rewards(
        json_path,
        tokenizer_name=base_model_name,
    )

    dataset = TensorDataset(input_ids, attention_mask, rewards)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Minimum 2 for std calculation

    print("Loading policy model...")
    policy = PolicyValueModel(base_model_name, adapter_path).to(device)
    
    # Enable gradient checkpointing to save memory
    policy.base.gradient_checkpointing_enable()
    
    # reference (KL anchor) = copy of initial policy (keep on CPU to save GPU memory)
    print("Loading reference model on CPU...")
    ref_policy = PolicyValueModel(base_model_name, adapter_path)
    ref_policy.load_state_dict(policy.state_dict(), strict=False)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)

    clip_range = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    kl_coef = 0.0   # set >0 if you want KL to reference

    num_epochs = 3
    for epoch in range(num_epochs):
        epoch_policy_losses = []
        epoch_value_losses = []
        
        for batch_idx, (batch_ids, batch_mask, batch_rewards) in enumerate(dataloader):
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_rewards = batch_rewards.to(device)

            # ---- Step 1: compute old logprobs & values (no grad) ----
            # runs on entire minibatch at once, this is the sampling for the minibatch loop
            with torch.no_grad():
                logits_old, values_old = policy(batch_ids, batch_mask)
                old_logprobs = compute_seq_logprobs(logits_old, batch_ids)  # [B]

            # One-step "returns" (no time dependence)
            # returns = rewards, advantage = rewards - V

            # Note that advantage is defined in the slides as: 
            # Advantage_t = Reward_(t+1) + gamma*value(S_t+1) - value(S_t)
            # Where value is evaluated using pi_theta, since pi_theta is our current policy, the value reduces down to
            # v(s) = values using the current policy
            returns = batch_rewards
            advantages = returns - values_old

            # normalize advantages (only if batch size > 1)
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            # Clip advantages to prevent extreme values
            advantages = torch.clamp(advantages, -10, 10)

            # ---- Step 2: PPO update ----
            logits_new, values_new = policy(batch_ids, batch_mask)
            new_logprobs = compute_seq_logprobs(logits_new, batch_ids)

            # prob ratio
            # since probabilities are logs, exp(logx - logy) = x / y = ratio

            ratio = torch.exp(new_logprobs - old_logprobs) 

            # clipped surrogate
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            # value loss
            value_loss = F.mse_loss(values_new, returns)

            # entropy bonus (encourage diversity)
            probs = torch.softmax(logits_new, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()

            # optional KL to ref model (compute on CPU to save GPU memory)
            kl = torch.tensor(0.0, device=device)
            if kl_coef > 0.0:
                with torch.no_grad():
                    batch_ids_cpu = batch_ids.cpu()
                    batch_mask_cpu = batch_mask.cpu()
                    ref_logits, _ = ref_policy(batch_ids_cpu, batch_mask_cpu)
                    ref_logprobs = compute_seq_logprobs(ref_logits, batch_ids_cpu)
                kl = (old_logprobs.cpu() - ref_logprobs).mean().to(device)

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

        avg_policy_loss = sum(epoch_policy_losses) / len(epoch_policy_losses) if epoch_policy_losses else float('nan')
        avg_value_loss = sum(epoch_value_losses) / len(epoch_value_losses) if epoch_value_losses else float('nan')
        
        print(
            f"Epoch {epoch}: "
            f"policy_loss={avg_policy_loss:.4f}, "
            f"value_loss={avg_value_loss:.4f}, "
            f"entropy={entropy.item():.4f}, "
            f"kl={kl.item():.4f}"
        )

    # Save PPO-updated model
    print("Saving PPO-tuned model...")
    policy.base.save_pretrained("ppo_tuned_model")  # Saves LoRA adapters
    torch.save(policy.value_head.state_dict(), "ppo_tuned_model/value_head.pt")  # Save value head
    print("Model saved to ppo_tuned_model/")


if __name__ == "__main__":
    main()
