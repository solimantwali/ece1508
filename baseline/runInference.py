import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from LMStudio import LMStudioClient
from loadConfig import load_mode_prompts

import time

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="JSON file containing either a single object or a list of objects with fields: id, prompt, (optional) mode_id",
    )

    p.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Where to write baseline outputs. Default: ./output/<input_filename>.txt",
    )

    p.add_argument("--mode_config", type=str, default="model-config.json",
                  help="JSON config mapping mode_id -> system prompt. "
                       "Default: model-config.json")

    p.add_argument("--lm_base_url", type=str, default="http://localhost:1234/v1",
                  help="LM Studio API base URL")

    p.add_argument("--model_name", type=str, default="llama-3-8b-instruct",
                  help="Model name as shown in LM Studio API")

    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=512)

    return p.parse_args()


def load_prompts(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    if isinstance(data, dict):
        return [data]

    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Input JSON list must contain objects.")
        return data

    raise ValueError("Input JSON must be either an object or a list of objects.")


def main() -> None:
    start = time.time()

    args = parse_args()

    # ----------- Handle defaults for output path -----------
    in_path = Path(args.input_json)

    if args.output_jsonl is None:
        # Strip the input extension and create output/<file_id>.txt
        file_id = in_path.stem
        output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{file_id}.txt"
    else:
        out_path = Path(args.output_jsonl)

    # ----------- Load mode → system prompt config -----------
    mode_to_system_prompt = load_mode_prompts(args.mode_config)

    # ----------- Instantiate LM Studio client -----------
    client = LMStudioClient(
        base_url=args.lm_base_url,
        model_name=args.model_name,
        default_temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    prompts = load_prompts(in_path)

    # ----------- Process input JSON -----------
    with out_path.open("w", encoding="utf-8") as fout:
        for ex in prompts:
            uid = ex.get("id")
            prompt = ex["prompt"]
            mode_id: Optional[int] = ex.get("mode_id")

            # Select system prompt if needed
            system_prompt = mode_to_system_prompt.get(mode_id)

            # LM Studio inference
            response = client.chat(prompt, system_prompt=system_prompt)

            out_obj = {
                "id": uid,
                "prompt": prompt,
                "mode_id": mode_id,
                "system_prompt": system_prompt,
                "model_name": args.model_name,
                "generation_config": {
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                },
                "response": response,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    end = time.time()
    elapsed = end - start
    print(f"Took {elapsed} seconds to run this inference. ")

if __name__ == "__main__":
    main()
