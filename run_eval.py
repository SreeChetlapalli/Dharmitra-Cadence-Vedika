#!/usr/bin/env python3
"""
Evaluate retrieval accuracy of the Dharmamitra search API.

Queries the API with (corrupted) segments and checks whether the correct
segmentnr appears in the top-k results. Reports Recall@1/5/10/50.

Usage:
    python run_eval.py [options]

Options:
    --samples-per-lang  N     Samples per language (default: 10)
    --languages         LANGS Comma-separated language codes (default: all)
    --corruption-levels LEVS  Comma-separated levels to test (default: 0,10,20,30,50)
    --corruption-types  TYPES Comma-separated types: none,crop,mask (default: all)
    --search-type       TYPE  semantic|fuzzy|combined (default: semantic)
    --do-ranking              Enable ranking (default: True)
    --no-ranking              Disable ranking
    --filter-lang             Pass filter_source_language=<lang> (default: False)
    --api-url           URL   API base URL
    --top-k             K     Maximum results to fetch (default: 50)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import urllib.request
import urllib.error

API_URL = "https://dharmamitra.org/api-search/primary/"
DATASET_PATH = Path(__file__).parent / "eval_dataset.jsonl"
TOP_KS = [1, 5, 10, 50]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-per-lang", type=int, default=10)
    p.add_argument("--languages", default=None, help="Comma-separated, e.g. bo,zh")
    p.add_argument("--corruption-levels", default=None, help="Comma-separated, e.g. 0,10,20")
    p.add_argument("--corruption-types", default=None, help="Comma-separated: none,crop,mask")
    p.add_argument("--search-type", default="semantic", choices=["semantic", "fuzzy", "combined"])
    p.add_argument("--do-ranking", dest="do_ranking", action="store_true", default=True)
    p.add_argument("--no-ranking", dest="do_ranking", action="store_false")
    p.add_argument("--filter-lang", action="store_true", default=False,
                   help="Pass filter_source_language matching query language")
    p.add_argument("--api-url", default=API_URL)
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def load_samples(args):
    """Load and filter eval samples, capped at samples_per_lang per language."""
    languages = set(args.languages.split(",")) if args.languages else None
    corruption_levels = set(int(x) for x in args.corruption_levels.split(",")) if args.corruption_levels else None
    corruption_types = set(args.corruption_types.split(",")) if args.corruption_types else None

    # Group by (language, segmentnr) then pick args.samples_per_lang unique segmentnrs per lang
    by_lang_seg = defaultdict(lambda: defaultdict(list))
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if languages and row["language"] not in languages:
                continue
            if corruption_levels is not None and row["corruption_level"] not in corruption_levels:
                continue
            if corruption_types and row["corruption_type"] not in corruption_types:
                continue
            by_lang_seg[row["language"]][row["segmentnr"]].append(row)

    samples = []
    for lang, seg_map in sorted(by_lang_seg.items()):
        segmentnrs = list(seg_map.keys())[:args.samples_per_lang]
        for segnr in segmentnrs:
            samples.extend(seg_map[segnr])
    return samples


def query_api(search_input, search_type, do_ranking, filter_lang, lang, api_url, top_k):
    payload = {
        "search_input": search_input,
        "search_type": search_type,
        "filter_source_language": lang if filter_lang else "all",
        "do_ranking": do_ranking,
        "page_size": top_k,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def hits_at_k(results, target_segmentnr, k):
    """Check if target_segmentnr appears in the top-k results."""
    for r in results[:k]:
        # results may have 'segmentnr' or 'all_segmentnrs'
        if r.get("segmentnr") == target_segmentnr:
            return True
        if target_segmentnr in r.get("all_segmentnrs", []):
            return True
    return False


def main():
    args = parse_args()
    samples = load_samples(args)

    if not samples:
        print("No samples matched the filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} samples")
    print(f"Settings: search_type={args.search_type}, do_ranking={args.do_ranking}, "
          f"filter_lang={args.filter_lang}, top_k={args.top_k}")
    print()

    # Stats: keyed by (language, corruption_type, corruption_level)
    # Each value: dict of k -> [hit, hit, ...]
    stats = defaultdict(lambda: defaultdict(list))

    for i, row in enumerate(samples, 1):
        lang = row["language"]
        level = row["corruption_level"]
        ctype = row["corruption_type"]
        key = (lang, ctype, level)

        print(f"[{i}/{len(samples)}] lang={lang} type={ctype} level={level} seg={row['segmentnr'][:40]}", end=" ", flush=True)

        try:
            results = query_api(
                search_input=row["corrupted"],
                search_type=args.search_type,
                do_ranking=args.do_ranking,
                filter_lang=args.filter_lang,
                lang=lang,
                api_url=args.api_url,
                top_k=args.top_k,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        for k in TOP_KS:
            hit = hits_at_k(results, row["segmentnr"], k)
            stats[key][k].append(hit)

        hit_summary = " ".join(f"@{k}={'Y' if stats[key][k][-1] else 'N'}" for k in TOP_KS)
        print(hit_summary)
        time.sleep(0.1)  # be polite

    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Aggregate by corruption level/type across all languages
    all_keys = sorted(stats.keys())

    header = f"{'Lang':<6} {'Type':<6} {'Level':>5}  " + "  ".join(f"R@{k:>2}" for k in TOP_KS) + "  N"
    print(header)
    print("-" * len(header))

    aggregate = defaultdict(lambda: defaultdict(list))  # (ctype, level) -> k -> hits

    for (lang, ctype, level) in all_keys:
        row_stats = stats[(lang, ctype, level)]
        n = len(row_stats[TOP_KS[0]])
        parts = []
        for k in TOP_KS:
            hits = row_stats[k]
            recall = sum(hits) / len(hits) if hits else 0.0
            parts.append(f"{recall:>5.1%}")
            aggregate[(ctype, level)][k].extend(hits)
        print(f"{lang:<6} {ctype:<6} {level:>5}  {'  '.join(parts)}  {n}")

    print()
    print("AGGREGATE (all languages)")
    print("-" * len(header))
    for (ctype, level) in sorted(aggregate.keys()):
        row_stats = aggregate[(ctype, level)]
        n = len(row_stats[TOP_KS[0]])
        parts = []
        for k in TOP_KS:
            hits = row_stats[k]
            recall = sum(hits) / len(hits) if hits else 0.0
            parts.append(f"{recall:>5.1%}")
        print(f"{'ALL':<6} {ctype:<6} {level:>5}  {'  '.join(parts)}  {n}")


if __name__ == "__main__":
    main()
