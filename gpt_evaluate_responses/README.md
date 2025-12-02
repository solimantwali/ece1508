This dir contains the code responsible for sending any model's responses to gpt-4o for sycophancy eval. Hasn't been tested.

OpenAi.py is meant to call the OpenAI API to get GPT-4o's responses. With a system prompt asking it to do the evaluation as we specify. 

Need a .env file with the OpenAPI key: 
OPENAI_API_KEY="your_openai_api_key_here"

Shard.py is meant to take an input json and split it into a number of jsons that we specify. This will create multiple files that we can then run through the OpenAI API in parallel to speed up the process. Tbh may not need this. 

