# config_utils.py
import json
from typing import Dict

def load_mode_prompts(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    mode_raw = raw.get("mode_to_system_prompt", {})
    return {int(k): v for k, v in mode_raw.items()}
