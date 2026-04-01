#!/usr/bin/env python3
"""
Extract uncorrupted Sanskrit passages from eval_dataset.jsonl into a
separate .jsonl file that preserves segment IDs for benchmarking.
"""

import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent / "eval_dataset.jsonl"
OUTPUT_PATH = Path(__file__).parent / "sanskrit_input.jsonl"


def main():
    if not INPUT_PATH.is_file():
        print(f"Error: {INPUT_PATH} not found.")
        return

    seen = set()
    with open(INPUT_PATH, encoding="utf-8") as f, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for line in f:
            row = json.loads(line)
            if row["language"] == "sa" and row["corruption_level"] == 0:
                key = row["segmentnr"]
                if key not in seen:
                    seen.add(key)
                    out.write(
                        json.dumps(
                            {"segmentnr": key, "original": row["original"]},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    print(f"Extracted {len(seen)} passages to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
