#!/usr/bin/env python3
"""
Evaluate retrieval accuracy of the Dharmamitra search API.
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
    p.add_argument("--corruption-types", default=None, help="Comma-separated, e.g. none,crop,mask or sentence,fixed_size,sliding_window,hierarchical")
    p.add_argument("--search-type", default="semantic", choices=["semantic", "fuzzy", "combined"])
    p.add_argument("--do-ranking", dest="do_ranking", action="store_true", default=True)
    p.add_argument("--no-ranking", dest="do_ranking", action="store_false")
    p.add_argument("--filter-lang", action="store_true", default=False,
                   help="Pass filter_source_language matching query language")
    p.add_argument("--api-url", default=API_URL)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--debug", action="store_true", help="Print request/response debugging info")
    return p.parse_args()


def load_samples(args):
    languages = set(args.languages.split(",")) if args.languages else None
    corruption_levels = set(int(x) for x in args.corruption_levels.split(",")) if args.corruption_levels else None
    corruption_types = set(args.corruption_types.split(",")) if args.corruption_types else None

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


def query_api(search_input, search_type, do_ranking, filter_lang, lang, api_url, top_k, debug=False):
    # Payload matching the public /primary/ schema
    payload = {
        "search_input": search_input,
        "input_encoding": "auto",
        "search_type": search_type,
        "semantic_type": "both",
        "filter_source_language": lang if filter_lang else "all",
        "filter_target_language": "all",
        "source_filters": {
            "include_files": [],
            "include_categories": [],
            "include_collections": [],
        },
        "do_ranking": do_ranking,
        "max_depth": top_k,
    }

    if debug:
        print(f"\nDEBUG request payload: {payload}", file=sys.stderr)

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e}")

    try:
        res_data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON response: {raw[:500]}")

    if debug:
        keys = list(res_data.keys()) if isinstance(res_data, dict) else None
        print(f"DEBUG response type: {type(res_data).__name__}, keys: {keys}", file=sys.stderr)

    if not isinstance(res_data, dict):
        raise RuntimeError(f"Unexpected top-level response type: {type(res_data).__name__}")

    if "results" not in res_data:
        raise RuntimeError(f"Missing 'results' in response: {res_data}")

    results = res_data["results"]

    if not isinstance(results, list):
        raise RuntimeError(f"'results' is not a list: {type(results).__name__}; response={res_data}")

    return results


def hits_at_k(results, target_segmentnr, k):
    for r in results[:k]:
        if not isinstance(r, dict):
            continue
        if r.get("segmentnr") == target_segmentnr:
            return True
        all_segmentnrs = r.get("all_segmentnrs", [])
        if isinstance(all_segmentnrs, list) and target_segmentnr in all_segmentnrs:
            return True
    return False


def main():
    args = parse_args()
    samples = load_samples(args)

    if not samples:
        print("No samples matched the filters.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} samples")
    print(
        f"Settings: search_type={args.search_type}, do_ranking={args.do_ranking}, "
        f"filter_lang={args.filter_lang}, top_k={args.top_k}"
    )
    print()

    # Use only cutoffs that are <= requested top_k
    ks = [k for k in TOP_KS if k <= args.top_k]
    if not ks:
        ks = [args.top_k]

    stats = defaultdict(lambda: defaultdict(list))
    failures = 0

    for i, row in enumerate(samples, 1):
        lang = row["language"]
        level = row["corruption_level"]
        ctype = row["corruption_type"]
        key = (lang, ctype, level)

        print(
            f"[{i}/{len(samples)}] lang={lang} type={ctype} level={level} seg={row['segmentnr'][:40]}",
            end=" ",
            flush=True
        )

        try:
            results = query_api(
                search_input=row["corrupted"],
                search_type=args.search_type,
                do_ranking=args.do_ranking,
                filter_lang=args.filter_lang,
                lang=lang,
                api_url=args.api_url,
                top_k=args.top_k,
                debug=args.debug,
            )
        except Exception as e:
            failures += 1
            print(f"ERROR: {e}")
            continue

        for k in ks:
            hit = hits_at_k(results, row["segmentnr"], k)
            stats[key][k].append(hit)

        hit_summary = " ".join(f"@{k}={'Y' if stats[key][k][-1] else 'N'}" for k in ks)
        print(hit_summary)
        time.sleep(0.1)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    all_keys = sorted(stats.keys())
    header = f"{'Lang':<6} {'Type':<6} {'Level':>5}  " + "  ".join(f"R@{k:>2}" for k in ks) + "  N"
    print(header)
    print("-" * len(header))

    aggregate = defaultdict(lambda: defaultdict(list))

    for (lang, ctype, level) in all_keys:
        row_stats = stats[(lang, ctype, level)]
        n = len(row_stats[ks[0]])
        parts = []
        for k in ks:
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
        n = len(row_stats[ks[0]])
        parts = []
        for k in ks:
            hits = row_stats[k]
            recall = sum(hits) / len(hits) if hits else 0.0
            parts.append(f"{recall:>5.1%}")
        print(f"{'ALL':<6} {ctype:<6} {level:>5}  {'  '.join(parts)}  {n}")

    print()
    print(f"Failures skipped: {failures}")


if __name__ == "__main__":
    main()