Need the oem_val_sft and oem_val_baseline files in this dir. 

Activate venv: 
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run in terminal:
EXPORT OPENAI_API_KEY=your_api_key_here

Then run the judge script to get the human, baseline, and sft model results after 4o judging. 
python judge.py

Then run the visualize script to get the bar graphs

Then run
python generate_rewards.py 
This will give you the final JSON to feed into PPO with the reward signals. The reward model is defined at the top of the file with the rows being the model's response from 0-3 and the cols being the human's response from 0-3. Reward is obtained using 2d index reward = reward_matrix[model_response][human_response].

Now feed this into the PPO code. 
