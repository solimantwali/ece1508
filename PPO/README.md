# PPO Training for Reducing Sycophancy

This directory contains the PPO (Proximal Policy Optimization) training code to reduce sycophancy in LLM outputs.

## Overview

We have a scalar reward system that uses the difference in sycophancy between the top Reddit response and our model's response. Using GPT4o as the judge, the judge code exists in the `../judge/` directory.

## Files

- `PolicyModel.py`: Defines the Policy and Value heads. Supports loading LoRA adapters.
- `ppo_train.py`: Main PPO training script.

## Setup

The code expects:
- Base model: `meta-llama/Meta-Llama-3-8B-Instruct`
- SFT adapter: `SFT_Model` (LoRA weights from SFT training)
- Rewards file: `oem_val_sft_with_rewards.json`

If you're on the lambdalabs instance, pip install -r requirements.txt
Or try to run the ppo_train.py and pip install whatever it tells you to. 

You may want to create a venv: 
python -m venv .venv
source .venv/bin/activate
Then do your pip installs here

export HF_TOKEN=your_huggingface_token_here


## Key Features

1. **Response-only training**: The PPO training operates directly on the model's responses (not prompts). The rewards are already assigned to these responses, so we simply tokenize the `model_response` field and use those tokens for policy optimization.

2. **LoRA support**: The PolicyValueModel can load LoRA adapters on top of the base model.

3. **Value head**: A scalar value head is added to estimate the expected reward (V function).

## Running PPO Training

```bash
cd /home/rohan/RLRepo/ece1508/PPO
python ppo_train.py
```

The trained model will be saved to `ppo_tuned_model/`.

## Configuration

In `ppo_train.py`, you can adjust:
- `batch_size`: Currently 4
- `num_epochs`: Currently 3
- `clip_range`: PPO clipping parameter (0.2)
- `value_coef`: Value loss coefficient (0.5)
- `entropy_coef`: Entropy bonus coefficient (0.01)
- `kl_coef`: KL divergence penalty (0.0, set >0 to enable)
- Learning rate: 1e-5 

## Testing
There is a test script: 
python sanity_test_model_is_updating.py 
Which runs the new models through some basic prompts. We do see that after PPO the model is changed, we haven't quantifiably measured this though to see if these changes are good. 