# ECE1508 - Reducing Sycophancy in LLMs

This repository contains implementations for training and evaluating language models to reduce sycophantic behavior using Supervised Fine-Tuning (SFT) and Proximal Policy Optimization (PPO).

PLEASE NOTE: We used AI for code generation and documentation in this repository. We would like to acknowledge the assistance of AI tools in the development process. Specifically Anthropic's Claude Sonnet and OpenAI's GPT-5 were used extensively in the coding of this project. 

Also note that this repo uses Huggingface's API to fetch the llama-3-8b model and OpenAI's API to use GPT-4o as a judge model. If you do not have access to the llama-3-8b model on HuggingFace or an OpenAI API key, you will not be able to run the scripts in this repo. Please reach out to Soliman Ali soliman.ali@mail.utoronto.ca or Vishnu Akundi vishnu.akundi@mail.utoronto.ca for access to the necessary resources. Please do NOT reach out to Rohan Patel rohanr.patel@mail.utoronto.ca as he will have dropped out of the university by the time this README is accessed (bye Ali, you taught a great course). 

## Repository Structure

### dataScripts/
Contains scripts and datasets for the data processing pipeline. All scripts have configurable parameters at the top of each file.

**Scripts:**
- `splitter.py` - Takes `oem_train_baseline.json` and splits it into three files: first 1k samples, second 1k samples, and remaining samples.
- `judge_responses.py` - Takes a JSON file with responses and uses GPT-4o to judge sycophancy levels (0-3 scale), outputs the same file with judge labels added.
- `generate_rewards.py` - Takes human-judged and model-judged files, computes rewards using a reward matrix, outputs a file with reward values for each entry.
- `create_bar_chart.py` - Takes multiple judged JSON files and generates a grouped bar chart comparing judge label distributions across models.

**Available Data for the SFT and PPO pipelines:**
- `oem_train_baseline.json` - Full training dataset (~2874 samples)
- `oem_train_baseline_first_1k.json` - First 1000 training samples
- `oem_train_baseline_1k_2k.json` - Samples 1000-1999
- `oem_train_baseline_2k_plus.json` - Samples 2000+
- `oem_val_baseline.json` - Validation dataset (~152 samples)

### SFT/
[To be documented]

### PPO/
[To be documented]

## Getting Started

1. Before running scripts in any repo, make sure to pip install the requirements.txt in that directory. For example in the dataScripts directory:
   ```bash
   cd dataScripts
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key for judging, this will make a query to GPT-4o for scoring the human/model responses to prompts:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. Set your HuggingFace token in terminal so that the llama-3-8b model can be pulled from HuggingFace: 
   ```bash
   export HF_TOKEN="your-huggingface-token-here"
   ```

- For any data cleaning/processing or running LLM as a judge or creating the bar chart which we used in our paper. Please go to the dataScripts directory and run the scripts as needed. Refer to the README in the dataScripts directory for more details.
