# Baseline Generation Pipeline (LM Studio)

This project runs a set of prompts through a local Llama-3 model served by LM Studio. The script reads a JSONL file of prompts, sends each prompt to the LM Studio API server, and writes the model outputs to a file for later analysis.

## Prerequisites

- Python 3.10 or later  
- LM Studio installed  
- A quantized Llama-3 model downloaded in LM Studio  
- LM Studio API server running at `http://localhost:1234/v1`

## Virtual Environment Setup

```
python -m venv .env
```

```
.\.env\Scripts\activate
```

```
pip install -r requirements.txt
```

If you need a minimal requirements file, create:

```
requests
```

## Running the LM Studio Server

1. Open LM Studio  
2. Load the desired model  
3. Start the API server from the LM Studio sidebar  
4. Confirm that the server URL displayed is `http://localhost:1234/v1`

Keep LM Studio open while you run the baseline script.

## Input Format

The input file must be JSONL. Each line should contain fields like:

```
{"id": "ex_01", "prompt": "Explain RLHF.", "mode_id": 0}
```

`mode_id` is optional. If present, it is used to select a system prompt.

## Mode Configuration

Create a file named `model-config.json`:

```
{
  "mode_to_system_prompt": {
    "0": "You are a default assistant.",
    "1": "You should challenge incorrect assumptions."
  }
}
```

If not provided, the script defaults to `model-config.json`.

## Running the Pipeline

Basic usage:

```
python run_baseline_lmstudio.py --input_jsonl data/prompts.jsonl
```

If no output path is provided, the script writes to:

```
output/<input_filename>.txt
```

Example with all parameters:

```
python run_baseline_lmstudio.py ^
  --input_jsonl data/prompts.jsonl ^
  --output_jsonl output/results.jsonl ^
  --mode_config model-config.json ^
  --model_name llama-3-8b-instruct ^
  --temperature 0.0 ^
  --max_tokens 512
```

## Output Format

The output file is JSONL. Each line contains:

- id  
- prompt  
- mode_id  
- selected system prompt  
- model name  
- generation settings  
- the model response

This file can be used directly in downstream analysis or comparison with RLHF-trained models.
