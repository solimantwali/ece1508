Don't worry about this file guys, this is my PoC to make sure I can run inference on multiple instances of a model in parallel. 

chat_template.txt may only be needed for models without a build in chat template. PoC uses gpt2 which does not have a chat template. 

source vllm-env/bin/activate
pip install -r requirements.txt

Starting server A: 
python -m vllm.entrypoints.openai.api_server \
  --model gpt2 \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 1024 \
  --chat-template '{% for message in messages %}{% if message["role"] == "user" %}{{ message["content"] }}{% endif %}{% endfor %}'

Starting server B:
python -m vllm.entrypoints.openai.api_server \
  --model gpt2 \
  --host 0.0.0.0 \
  --port 8002 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 1024 \
  --chat-template '{% for message in messages %}{% if message["role"] == "user" %}{{ message["content"] }}{% endif %}{% endfor %}'

Then try:
python parallel_client.py 

The config.json will default to InputData/smallTest/in_config.json

3 Different terminals. 

May need to set your huggingface token:
export HF_TOKEN="hf_your_token_here"

Make sure to change parallel_client.py for the meta model: 
"model": "meta-llama/Meta-Llama-3-8B-Instruct"
and change:
            "max_tokens": 64,
To: 
            "max_tokens": 1024,

For the non PoC we can use (discarding the chat template):
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8002 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 4096
