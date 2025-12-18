# models.py
# Defines the Policy and Value heads. The policy head is used to grab the output logits
# Value head is used to estimate the reward
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig


class PolicyValueModel(nn.Module):
    """
    Wraps a causal LM and adds a scalar value head.
    Supports loading a base model with LoRA adapters.
    """
    def __init__(self, base_model_name: str, adapter_path: str = None, dtype: torch.dtype = torch.bfloat16, lora_config: LoraConfig=None):
        super().__init__()
        self.base = AutoModelForCausalLM.from_pretrained(base_model_name, dtype=dtype)
        
        # Load LoRA adapters if provided
        if adapter_path is not None:
            self.base = PeftModel.from_pretrained(self.base, adapter_path)

        elif lora_config is not None:
            self.base = get_peft_model(self.base, lora_config)

        hidden_size = self.base.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(dtype=dtype)


    def forward(self, input_ids, attention_mask=None):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Last hidden layer [B, T, H]
        h_last = out.hidden_states[-1]
        # You can choose last token, or some pooling; last token is simple:
        h_last_token = h_last[:, -1, :]  # [B, H]

        values = self.value_head(h_last_token).squeeze(-1)  # [B]
        logits = out.logits  # [B, T, V]
        return logits, values 
        # think of the returns as:
        # (policy_head, value_head) 


def compute_seq_logprobs(logits, input_ids, mask=None, average=False):
    """
    Compute log prob of the *given* sequence of tokens.
    logits: [B, T, V]
    input_ids: [B, T]
    mask: [B, T] 1 for tokens we want to include in the sum (e.g. response),
          or None to use all tokens.
    Returns: [B] logprob per sequence.
    """
    # log_probs = F.log_softmax(logits, dim=-1)           # [B, T, V]
    # token_log_probs = log_probs.gather(
    #     dim=-1, index=input_ids.unsqueeze(-1)
    # ).squeeze(-1)                                       # [B, T]

    B, T, V = logits.shape

    # Flatten for CE
    logits_flat = logits.view(-1, V)          # [B*T, V]
    targets_flat = input_ids.view(-1)         # [B*T]

    # CE: fused log-softmax + NLL, no [B,T,V] log_probs stored
    nll_flat = F.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
    )  # [B*T]

    token_log_probs = -nll_flat.view(B, T)    # [B, T]

    if mask is not None:
        token_log_probs = token_log_probs * mask
        lengths = mask.sum(dim=-1).clamp(min=1)
        # For PPO we usually sum over response tokens
        seq_logprobs = token_log_probs.sum(dim=-1)
        if average:
            seq_logprobs = seq_logprobs / lengths
    else:
        if average:
            seq_logprobs = token_log_probs.mean(dim=-1)   # per-token average
        else:
            seq_logprobs = token_log_probs.sum(dim=-1)

    return seq_logprobs  # [B]
