import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path

def get_lora_params(model):
    """Return a dict {name: tensor.clone()} for all LoRA parameters."""
    lora = {}
    for name, p in model.named_parameters():
        if "lora_" in name:
            if getattr(p, "is_meta", False):
                continue
            lora[name] = p.detach().cpu().clone()
    return lora


def load_peft(base, adapter):
    """Load a PEFT model with LoRA adapters."""
    model = AutoModelForCausalLM.from_pretrained(base)
    model = PeftModel.from_pretrained(model, adapter)
    return model


def main():

    BASE = "meta-llama/Meta-Llama-3-8B-Instruct"

    ORIGINAL = "SFT_Model"          # your supervised fine-tuned LoRA adapters
    PPO = "ppo_tuned_model"         # your PPO-updated LoRA adapters

    print("Loading original SFT model LoRA...")
    model_orig = load_peft(BASE, ORIGINAL)
    orig_lora = get_lora_params(model_orig)

    print("Loading PPO-tuned model LoRA...")
    model_ppo = load_peft(BASE, PPO)
    ppo_lora = get_lora_params(model_ppo)

    print("\n=== Measuring LoRA Parameter Differences ===")
    diffs = []

    for name in orig_lora:
        if name not in ppo_lora:
            print(f"WARNING: Missing {name} in PPO model!")
            continue

        o = orig_lora[name]
        p = ppo_lora[name]
        delta = (p - o)

        diff_mean = delta.abs().mean().item()
        diff_max = delta.abs().max().item()
        diff_norm = delta.norm().item()

        diffs.append((name, diff_mean, diff_max, diff_norm))

    # Sort by largest norm change
    diffs.sort(key=lambda x: x[3], reverse=True)

    for name, dmean, dmax, dnorm in diffs:
        print(
            f"\n{name}\n"
            f"  mean |Δ| = {dmean:.6e}\n"
            f"  max  |Δ| = {dmax:.6e}\n"
            f"  L2 norm Δ = {dnorm:.6e}"
        )

    print("\nDone. If all numbers are ~0, PPO did NOT update your LoRA weights.")


if __name__ == "__main__":
    main()
