# save as estimate_model_memory.py and run: python estimate_model_memory.py facebook/llama-7b --dtype fp16 --max-seq 512
import sys
from math import ceil

def bytes_per_param(dtype):
    if dtype in ("fp32","float32","float"):
        return 4
    if dtype in ("fp16","half","bf16","bfloat16"):
        return 2
    if dtype in ("q4","4bit","int4"):
        return 0.5
    return 2

def estimate_from_config(H, L, dtype="fp16", max_seq=512, workspace_gb=0.5, overhead_frac=0.10):
    bp = bytes_per_param(dtype)
    num_params_est = 12 * (H**2) * L   # approximate
    weights_bytes = num_params_est * bp
    kv_per_token_bytes = 2 * H * L * bp
    kv_total_bytes = kv_per_token_bytes * max_seq
    base = weights_bytes + kv_total_bytes + (workspace_gb * 1024**3)
    total = base * (1 + overhead_frac)
    return {
        "num_params_est": int(num_params_est),
        "weights_GB": weights_bytes/1024**3,
        "kv_per_token_MB": kv_per_token_bytes/1024**2,
        "kv_for_max_seq_GB": kv_total_bytes/1024**3,
        "total_est_GB": total/1024**3,
    }

if __name__ == "__main__":
    # If you have HF transformers and want to auto-get H,L:
    try:
        from transformers import AutoConfig
    except Exception:
        AutoConfig = None

    if len(sys.argv) < 2:
        print("Usage: python estimate_model_memory.py <model-or-config-name> [--dtype fp16] [--max-seq 512]")
        sys.exit(1)

    model = sys.argv[1]
    dtype = "fp16"
    max_seq = 512
    if "--dtype" in sys.argv:
        dtype = sys.argv[sys.argv.index("--dtype")+1]
    if "--max-seq" in sys.argv:
        max_seq = int(sys.argv[sys.argv.index("--max-seq")+1])

    if AutoConfig:
        try:
            cfg = AutoConfig.from_pretrained(model, trust_remote_code=False)
            H = getattr(cfg, "hidden_size", None)
            L = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
            print("Config read from", model)
        except Exception as e:
            print("Could not load config from HF:", e)
            H = L = None
    else:
        print("transformers not available; please provide H and L manually")
        H = L = None

    if H is None or L is None:
        # fallback values for LLaMA-7B
        print("Using example LLaMA-7B defaults (H=4096, L=32).")
        H = 4096
        L = 32

    out = estimate_from_config(H, L, dtype=dtype, max_seq=max_seq)
    print("Estimated params:", out["num_params_est"])
    print(f"Weights (GB) ~ {out['weights_GB']:.2f} GB ({dtype})")
    print(f"KV per token ~ {out['kv_per_token_MB']:.2f} MB")
    print(f"KV for {max_seq} tokens ~ {out['kv_for_max_seq_GB']:.2f} GB")
    print(f"Total estimated GPU per-instance ~ {out['total_est_GB']:.2f} GB (including overhead)")
    