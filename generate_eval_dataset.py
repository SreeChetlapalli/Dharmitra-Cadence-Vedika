#!/usr/bin/env python3
"""
Generate evaluation dataset for retrieval system testing.

Samples 1000 segments per language, applies corruption at multiple levels
using two strategies: crop (random contiguous window) and mask (random span removal).

Output: eval_dataset.jsonl with fields:
  language, segmentnr, original, corrupted, corruption_level, corruption_type
  corruption_level 0 = no corruption (original)
"""

import json
import random
from pathlib import Path

LANGUAGES = {
    "bo": "/Users/snehrdich/data/dharmanexus-tibetan/segments",
    "zh": "/Users/snehrdich/data/dharmanexus-chinese/segments",
    "pa": "/Users/snehrdich/data/dharmanexus-pali/segments",
    "sa": "/Users/snehrdich/data/dharmanexus-sanskrit/segments",
}

CORRUPTION_LEVELS = [10, 20, 30, 50]
SAMPLE_SIZE = 1000
MIN_LENGTH = 30  # skip segments shorter than this (not useful for retrieval testing)
SEED = 42


def iter_segments(segments_dir: str):
    """Yield (segmentnr, original) from all JSON files in the directory."""
    for path in sorted(Path(segments_dir).glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for seg in data:
                original = (seg.get("original") or "").strip()
                if len(original) >= MIN_LENGTH:
                    yield {"segmentnr": seg["segmentnr"], "original": original}
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
            continue


def corrupt_crop(text: str, level: int, rng: random.Random) -> str:
    """
    Keep a random contiguous window of (100 - level)% of the text.
    e.g. at level=20, keep 80% of chars from a random start position.
    """
    keep = max(1, int(len(text) * (100 - level) / 100))
    max_start = len(text) - keep
    start = rng.randint(0, max_start)
    return text[start:start + keep].strip()


def corrupt_mask(text: str, level: int, rng: random.Random) -> str:
    """
    Remove random contiguous spans totalling ~level% of the text.
    Spans are deleted (not replaced) to simulate partial/damaged text.
    """
    target_removed = max(1, int(len(text) * level / 100))
    chars = list(text)
    removed = 0
    attempts = 0

    while removed < target_removed and attempts < 100:
        attempts += 1
        remaining_to_remove = target_removed - removed
        # span length between 1 and up to 1/4 of text, but at most what's left to remove
        max_span = max(1, min(remaining_to_remove, len(chars) // 4))
        span_len = rng.randint(1, max_span)
        # pick a start that keeps span in bounds; avoid already-emptied chars
        non_empty = [i for i, c in enumerate(chars) if c]
        if len(non_empty) < span_len:
            break
        anchor = rng.choice(non_empty)
        start = max(0, min(anchor, len(chars) - span_len))
        for i in range(start, start + span_len):
            if chars[i]:
                removed += 1
                chars[i] = ""

    return "".join(chars).strip()


def main():
    rng = random.Random(SEED)
    output_path = Path("/Users/snehrdich/data/dharmanexus-evaluation/eval_dataset.jsonl")

    total_rows = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for lang, segments_dir in LANGUAGES.items():
            print(f"\nProcessing {lang} ({segments_dir}) ...")
            all_segments = list(iter_segments(segments_dir))
            print(f"  Total valid segments: {len(all_segments)}")

            if len(all_segments) < SAMPLE_SIZE:
                print(f"  WARNING: only {len(all_segments)} segments available, using all")

            sample = rng.sample(all_segments, min(SAMPLE_SIZE, len(all_segments)))
            print(f"  Sampled: {len(sample)}")

            for seg in sample:
                base = {
                    "language": lang,
                    "segmentnr": seg["segmentnr"],
                    "original": seg["original"],
                }

                # Level 0: uncorrupted baseline
                out.write(json.dumps({
                    **base,
                    "corrupted": seg["original"],
                    "corruption_level": 0,
                    "corruption_type": "none",
                }, ensure_ascii=False) + "\n")

                for level in CORRUPTION_LEVELS:
                    # Crop: random contiguous window
                    out.write(json.dumps({
                        **base,
                        "corrupted": corrupt_crop(seg["original"], level, rng),
                        "corruption_level": level,
                        "corruption_type": "crop",
                    }, ensure_ascii=False) + "\n")

                    # Mask: random span removal
                    out.write(json.dumps({
                        **base,
                        "corrupted": corrupt_mask(seg["original"], level, rng),
                        "corruption_level": level,
                        "corruption_type": "mask",
                    }, ensure_ascii=False) + "\n")

            rows_for_lang = len(sample) * (1 + len(CORRUPTION_LEVELS) * 2)
            total_rows += rows_for_lang
            print(f"  Wrote {rows_for_lang} rows for {lang}")

    print(f"\nDone. Total rows: {total_rows}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
