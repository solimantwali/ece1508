#
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from models import PolicyValueModel, compute_seq_logprobs


def load_dataset_with_rewards(json_path, tokenizer_name, max_length=512):
    """
    JSON format:
    [
      { "prompt": "...", "response": "...", "reward": 0.7 },
      ...
    ]
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    prompts = [d["prompt"] for d in data]
    responses = [d["response"] for d in data]
    rewards = torch.tensor([d["reward"] for d in data], dtype=torch.float32)

    # concat prompt + EOS + response
    eos = tokenizer.eos_token or ""
    texts = [p + eos + r for p, r in zip(prompts, responses)]

    enc = tokenizer(
        texts,
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

    base_model_name = "gpt2"  # replace with your SFT model
    json_path = "rewards_dataset.json"

    input_ids, attention_mask, rewards = load_dataset_with_rewards(
        json_path,
        tokenizer_name=base_model_name,
    )

    dataset = TensorDataset(input_ids, attention_mask, rewards)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    policy = PolicyValueModel(base_model_name).to(device)
    # reference (KL anchor) = copy of initial policy
    ref_policy = PolicyValueModel(base_model_name).to(device)
    ref_policy.load_state_dict(policy.state_dict())
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
        for batch_ids, batch_mask, batch_rewards in dataloader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_rewards = batch_rewards.to(device)

            # ---- Step 1: compute old logprobs & values (no grad) ----
            # runs on entire minibatch at once, this is the sampling for the minibatch loop
            with torch.no_grad():
                logits_old, values_old = policy(batch_ids, batch_mask)
                old_logprobs = compute_seq_logprobs(logits_old, batch_ids)  # [B]

            # One-step “returns” (no time dependence)
            # returns = rewards, advantage = rewards - V

            # Note that advantage is defined in the slides as: 
            # Advantage_t = Reward_(t+1) + gamma*value(S_t+1) - value(S_t)
            # Where value is evaluated using pi_theta, since pi_theta is our current policy, the value reduces down to
            # v(s) = values using the current policy
            returns = batch_rewards
            advantages = returns - values_old

            # normalize advantages, come back to this
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ---- Step 2: PPO update ----
            logits_new, values_new = policy(batch_ids, batch_mask)
            new_logprobs = compute_seq_logprobs(logits_new, batch_ids)

            # prob ratio
            ratio = torch.exp(new_logprobs - old_logprobs)  # [B]

            # clipped surrogate
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.mean(torch.min(unclipped, clipped))

            # value loss
            value_loss = F.mse_loss(values_new, returns)

            # entropy bonus (encourage diversity)
            probs = torch.softmax(logits_new, dim=-1)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()

            # optional KL to ref model
            kl = torch.tensor(0.0, device=device)
            if kl_coef > 0.0:
                with torch.no_grad():
                    ref_logits, _ = ref_policy(batch_ids, batch_mask)
                    ref_logprobs = compute_seq_logprobs(ref_logits, batch_ids)
                kl = (old_logprobs - ref_logprobs).mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + kl_coef * kl
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

        print(
            f"Epoch {epoch}: "
            f"policy_loss={policy_loss.item():.4f}, "
            f"value_loss={value_loss.item():.4f}, "
            f"entropy={entropy.item():.4f}, "
            f"kl={kl.item():.4f}"
        )

    # Save PPO-updated model
    policy.base.save_pretrained("ppo_tuned_model")


if __name__ == "__main__":
    main()
