#!/usr/bin/env python
"""
Shard a JSON array into N separate JSON files.

Usage:
    python shard_json.py input.json 5

This will create:
    sharded/shard_000.json
    sharded/shard_001.json
    ...
    sharded/shard_004.json
"""

import json
import math
import sys
from pathlib import Path
from typing import List, Any


def load_json_array(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array, got {type(data)}")

    return data


def shard_array(data: List[Any], num_shards: int) -> List[List[Any]]:
    """
    Split `data` into num_shards chunks, preserving order and distributing
    elements as evenly as possible.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    n = len(data)
    if n == 0:
        # All shards will be empty
        return [[] for _ in range(num_shards)]

    shard_sizes = []
    base = n // num_shards
    remainder = n % num_shards

    for i in range(num_shards):
        size = base + (1 if i < remainder else 0)
        shard_sizes.append(size)

    shards = []
    idx = 0
    for size in shard_sizes:
        shard = data[idx: idx + size]
        shards.append(shard)
        idx += size

    return shards


def save_shards(
    shards: List[List[Any]],
    out_dir: Path,
    base_name: str = "shard",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, shard in enumerate(shards):
        out_path = out_dir / f"{base_name}_{i:03d}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(shard, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(shard)} items to {out_path}")


def main(argv: List[str]) -> None:
    if len(argv) != 3:
        print("Usage: python shard_json.py <input_json> <num_shards>")
        sys.exit(1)

    input_path = Path(argv[1])
    num_shards = int(argv[2])

    if not input_path.exists():
        print(f"Error: input file '{input_path}' does not exist")
        sys.exit(1)

    print(f"Loading JSON from {input_path}")
    data = load_json_array(input_path)
    print(f"Total items: {len(data)}")

    print(f"Sharding into {num_shards} files...")
    shards = shard_array(data, num_shards)

    out_dir = Path("sharded")
    save_shards(shards, out_dir)


if __name__ == "__main__":
    main(sys.argv)
