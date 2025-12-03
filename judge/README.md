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


