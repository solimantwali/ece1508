from peft import PeftModel
from transformers import AutoModelForCausalLM

# BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# ADAPTER_DIR = "mistral7b_oem_sft"

BASE_MODEL  = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_DIR = "llama3_8b_oem_sft"

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
sft  = PeftModel.from_pretrained(base, ADAPTER_DIR)

count = 0
for name, param in sft.named_parameters():
    if "lora" in name:
        print(name, "mean |w| =", param.data.abs().mean().item())
        count += 1
        if count >= 5:  # just print a few
            break

print("Total LoRA params found:", count)
