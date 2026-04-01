#!/usr/bin/env python3
"""
Sanskrit Chunking Pipeline
==========================

Normalize -> Punctuate -> Chunk Sanskrit text into retrieval-ready segments.

Usage:
    python sanskrit_pipeline.py --segments-dir ./path/to/segments --strategy sentence
    python sanskrit_pipeline.py --input-file sample.txt --strategy sliding_window
    python sanskrit_pipeline.py --demo
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Vedika imports — the package __init__.py unconditionally downloads model
# weights on import, which fails when the network is flaky.  The normalizer
# and sentence splitter are pure rule-based and don't need those weights, so
# we load their .py files directly via importlib.
# ---------------------------------------------------------------------------
VEDIKA_AVAILABLE = False
_vedika_normalizer_mod = None
_vedika_splitter_mod = None

try:
    import importlib.util as _ilu

    _vedika_pkg = _ilu.find_spec("vedika")
    if _vedika_pkg and _vedika_pkg.submodule_search_locations:
        _vedika_root = _vedika_pkg.submodule_search_locations[0]

        _sp_path = Path(_vedika_root) / "sentence_splitter.py"
        if _sp_path.exists():
            _sp_spec = _ilu.spec_from_file_location(
                "vedika_sentence_splitter", str(_sp_path)
            )
            _vedika_splitter_mod = _ilu.module_from_spec(_sp_spec)
            _sp_spec.loader.exec_module(_vedika_splitter_mod)

        _nm_path = Path(_vedika_root) / "normalizer.py"
        if _nm_path.exists():
            _nm_spec = _ilu.spec_from_file_location(
                "vedika_normalizer", str(_nm_path)
            )
            _vedika_normalizer_mod = _ilu.module_from_spec(_nm_spec)
            _nm_spec.loader.exec_module(_vedika_normalizer_mod)

        VEDIKA_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Cadence import
# ---------------------------------------------------------------------------
try:
    from cadence import PunctuationModel

    CADENCE_AVAILABLE = True
except ImportError:
    CADENCE_AVAILABLE = False


# ============================= Data Loading ================================


def load_segments_from_dir(
    segments_dir: str, min_length: int = 30
) -> List[dict]:
    """Load {"segmentnr", "original"} dicts from DharmaNexus-style JSON files."""
    segments_path = Path(segments_dir)
    if not segments_path.is_dir():
        print(f"Error: segments directory not found: {segments_dir}", file=sys.stderr)
        sys.exit(1)

    segments: list[dict] = []
    for path in sorted(segments_path.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for seg in data:
                text = (seg.get("original") or "").strip()
                if len(text) >= min_length:
                    segments.append(
                        {"segmentnr": seg["segmentnr"], "original": text}
                    )
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
    return segments


def load_segments_from_txt(txt_path: str, min_length: int = 30) -> List[dict]:
    """Load raw UTF-8 text file -- each non-blank line becomes a segment."""
    path = Path(txt_path)
    if not path.is_file():
        print(f"Error: file not found: {txt_path}", file=sys.stderr)
        sys.exit(1)

    segments: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.strip()
            if len(text) >= min_length:
                segments.append(
                    {"segmentnr": f"{path.stem}:{i}", "original": text}
                )
    return segments


def load_segments_from_jsonl(jsonl_path: str, min_length: int = 30) -> List[dict]:
    """
    Load segments from a .jsonl file, preserving the original segmentnr.

    Each line must be a JSON object with at least an "original" field.
    If "segmentnr" is present it is kept; otherwise a synthetic one is assigned.
    Rows can optionally be filtered to a single language via the "language" field.
    """
    path = Path(jsonl_path)
    if not path.is_file():
        print(f"Error: file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    segments: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("original") or "").strip()
            if len(text) < min_length:
                continue
            segments.append({
                "segmentnr": row.get("segmentnr", f"{path.stem}:{i}"),
                "original": text,
            })
    return segments


# =========================== Normalisation =================================


def normalize_text(text: str) -> str:
    """Normalize Sanskrit text using Vedika (character-level cleanup only)."""
    if VEDIKA_AVAILABLE and _vedika_normalizer_mod is not None:
        return _vedika_normalizer_mod.normalize_standard_sanskrit_text(text)
    import unicodedata

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================ Punctuation ==================================


_cadence_model = None


def get_cadence_model(model_name: str = "Cadence-Fast", cpu: bool = True):
    """Lazy-load the Cadence model so startup is fast when --help is used."""
    global _cadence_model
    if _cadence_model is None:
        if not CADENCE_AVAILABLE:
            print(
                "Error: cadence-punctuation is not installed. "
                "Run: pip install cadence-punctuation",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loading {model_name} model ...")
        t0 = time.time()
        _cadence_model = PunctuationModel(
            model=model_name,
            sliding_window=True,
            max_length=300,
            cpu=cpu,
        )
        print(f"Model loaded in {time.time() - t0:.1f}s")
    return _cadence_model


def add_punctuation(
    texts: List[str],
    model_name: str = "Cadence-Fast",
    batch_size: int = 8,
    cpu: bool = True,
) -> List[str]:
    """Restore punctuation to a batch of Sanskrit texts via Cadence."""
    model = get_cadence_model(model_name=model_name, cpu=cpu)
    return model.punctuate(texts, batch_size=batch_size)


# ========================= Sentence Splitting ==============================


def split_sanskrit_sentences(text: str) -> List[str]:
    """
    Split punctuated Sanskrit text into sentences.

    Priority:
      1. Danda / double-danda
      2. Vedika SentenceSplitter (if available)
      3. Latin punctuation fallback (for romanised IAST texts)
    """
    sentences = re.split(r"[।॥]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) > 1:
        return sentences

    if VEDIKA_AVAILABLE and _vedika_splitter_mod is not None:
        try:
            result = _vedika_splitter_mod.split_sentences(text)
            if len(result) > 1:
                return result
        except Exception:
            pass

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# ============================ Chunking =====================================
#
# Each chunk_* function takes a text and parameters, and returns a list of
# clean text chunks.  No corruption or degradation -- just splitting.


def chunk_sentence(text: str, **_kwargs) -> List[str]:
    """
    Split at sentence boundaries (dandas).
    Each sentence becomes its own chunk.
    """
    sentences = split_sanskrit_sentences(text)
    if not sentences:
        return [text] if text.strip() else []
    return sentences


def chunk_fixed_size(text: str, chunk_size: int = 50, **_kwargs) -> List[str]:
    """
    Split into chunks of approximately chunk_size words.
    Tries to break at the nearest sentence boundary to avoid cutting
    mid-sentence.
    """
    sentences = split_sanskrit_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        if current_words and len(current_words) + len(words) > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = []
        current_words.extend(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_sliding_window(
    text: str, chunk_size: int = 50, overlap: float = 0.25, **_kwargs
) -> List[str]:
    """
    Overlapping word-windows of chunk_size words.
    Every part of the text appears in at least one chunk; boundaries
    are covered by the overlap region.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text.strip()]

    overlap_words = max(1, int(chunk_size * overlap))
    step = max(1, chunk_size - overlap_words)

    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break

    return chunks


def chunk_hierarchical(text: str, **_kwargs) -> List[str]:
    """
    Produce chunks at multiple granularity levels from the same text:
      Level 1 -- individual sentences
      Level 2 -- pairs of consecutive sentences
      Level 3 -- groups of 3+ sentences (verse-level)
    All levels are included in the output.
    """
    sentences = split_sanskrit_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    level1 = list(sentences)

    level2 = [
        " ".join(sentences[i : i + 2])
        for i in range(0, len(sentences) - 1)
    ]

    level3 = []
    if len(sentences) >= 3:
        level3 = [
            " ".join(sentences[i : i + 3])
            for i in range(0, len(sentences) - 2)
        ]

    return level1 + level2 + level3


STRATEGIES = {
    "sentence": chunk_sentence,
    "fixed_size": chunk_fixed_size,
    "sliding_window": chunk_sliding_window,
    "hierarchical": chunk_hierarchical,
}


# ========================== Pipeline Runner ================================


def run_pipeline(
    segments: List[dict],
    strategy: str = "sentence",
    chunk_size: int = 50,
    overlap: float = 0.25,
    model_name: str = "Cadence-Fast",
    batch_size: int = 8,
    cpu: bool = True,
    skip_punctuation: bool = False,
) -> List[dict]:
    """
    Full pipeline: Normalize -> Punctuate -> Chunk.

    Returns a flat list of chunk rows, each containing:
      segmentnr, original (full passage), chunk (one piece of it),
      strategy, chunk_size.
    """
    chunk_fn = STRATEGIES.get(strategy, chunk_sentence)

    # --- Normalize ---
    print("Normalizing ...")
    for seg in segments:
        seg["normalized"] = normalize_text(seg["original"])

    # --- Punctuate ---
    if skip_punctuation:
        print("Skipping punctuation (--skip-punctuation)")
        for seg in segments:
            seg["punctuated"] = seg["normalized"]
    else:
        print(f"Adding punctuation with {model_name} (batch_size={batch_size}) ...")
        texts = [seg["normalized"] for seg in segments]
        punctuated = add_punctuation(
            texts, model_name=model_name, batch_size=batch_size, cpu=cpu
        )
        for seg, punc_text in zip(segments, punctuated):
            seg["punctuated"] = punc_text

    # --- Chunk ---
    print(f"Chunking with strategy={strategy}, chunk_size={chunk_size} ...")
    rows: list[dict] = []
    for seg in segments:
        chunks = chunk_fn(
            seg["punctuated"], chunk_size=chunk_size, overlap=overlap
        )
        for i, chunk_text in enumerate(chunks):
            rows.append({
                "segmentnr": seg["segmentnr"],
                "original": seg["original"],
                "punctuated": seg["punctuated"],
                "chunk": chunk_text,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "strategy": strategy,
                "chunk_size": chunk_size,
            })

    print(f"  {len(segments)} segments -> {len(rows)} chunks")
    return rows


# ============================ Output =======================================


def save_for_eval(
    rows: List[dict],
    output_path: str,
    strategy: str,
    chunk_size: int,
):
    """Write eval_dataset.jsonl compatible with run_eval.py."""
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            eval_row = {
                "language": "sa",
                "segmentnr": row["segmentnr"],
                "original": row["original"],
                "corrupted": row["chunk"],
                "corruption_level": chunk_size,
                "corruption_type": strategy,
            }
            f.write(json.dumps(eval_row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows -> {output_path}")


def print_samples(rows: List[dict], n: int = 5):
    """Pretty-print a few chunk results for quick inspection."""
    seen_segments = set()
    shown = 0
    for row in rows:
        seg_id = row["segmentnr"]
        if seg_id not in seen_segments:
            seen_segments.add(seg_id)
            print("-" * 72)
            original_display = row["original"][:100]
            if len(row["original"]) > 100:
                original_display += " ..."
            print(f"  Segment:   {seg_id}")
            print(f"  Original:  {original_display}")
            if row.get("punctuated") and row["punctuated"] != row["original"]:
                punc_display = row["punctuated"][:100]
                if len(row["punctuated"]) > 100:
                    punc_display += " ..."
                print(f"  Punctuated: {punc_display}")
            print(f"  Chunks ({row['total_chunks']}):")

        chunk_display = row["chunk"][:90]
        if len(row["chunk"]) > 90:
            chunk_display += " ..."
        print(f"    [{row['chunk_index']+1}/{row['total_chunks']}] {chunk_display}")
        shown += 1

        if len(seen_segments) >= n and row["chunk_index"] == row["total_chunks"] - 1:
            break
    print("-" * 72)


# ================================ Demo =====================================


DEMO_TEXTS = [
    "धर्मो रक्षति रक्षितः। सत्यं वद। धर्मं चर। स्वाध्यायान्मा प्रमदः।",
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत। अभ्युत्थानमधर्मस्य तदात्मानं सृजाम्यहम्। परित्राणाय साधूनां विनाशाय च दुष्कृताम्। धर्मसंस्थापनार्थाय सम्भवामि युगे युगे।",
    "असतो मा सद्गमय। तमसो मा ज्योतिर्गमय। मृत्योर्मा अमृतं गमय।",
    "सर्वे भवन्तु सुखिनः। सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु। मा कश्चिद्दुःखभाग्भवेत्।",
    "विद्या ददाति विनयम्। विनयाद्याति पात्रताम्। पात्रत्वाद्धनमाप्नोति। धनाद्धर्मं ततः सुखम्।",
]


def run_demo(args):
    """Run the pipeline on built-in sample Sanskrit texts."""
    print("=" * 72)
    print("  Sanskrit Chunking Pipeline -- Demo")
    print("=" * 72)

    segments = [
        {"segmentnr": f"demo:{i+1}", "original": t}
        for i, t in enumerate(DEMO_TEXTS)
    ]

    rows = run_pipeline(
        segments,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model_name=args.model,
        batch_size=args.batch_size,
        cpu=args.cpu,
        skip_punctuation=args.skip_punctuation,
    )

    print()
    print_samples(rows, n=len(segments))

    if args.output:
        save_for_eval(rows, args.output, args.strategy, args.chunk_size)


# ================================= CLI =====================================


def parse_args():
    p = argparse.ArgumentParser(
        description="Sanskrit Chunking Pipeline: Normalize -> Punctuate -> Chunk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--segments-dir",
        help="Path to directory of DharmaNexus-style .json segment files",
    )
    src.add_argument(
        "--input-file",
        help="Path to a plain UTF-8 .txt file (one passage per line)",
    )
    src.add_argument(
        "--jsonl-input",
        help="Path to a .jsonl file with segmentnr + original fields (preserves segmentnrs for benchmarking)",
    )
    src.add_argument(
        "--demo",
        action="store_true",
        help="Run on built-in sample Sanskrit texts",
    )

    p.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="sentence",
        help="Chunking strategy (default: sentence)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Target chunk size in words for fixed_size/sliding_window (default: 50)",
    )
    p.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap fraction for sliding_window, 0.0-1.0 (default: 0.25)",
    )

    p.add_argument(
        "--model",
        default="Cadence-Fast",
        choices=["Cadence", "Cadence-Fast"],
        help="Cadence model variant (default: Cadence-Fast)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for Cadence punctuation (default: 8)",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        default=True,
        help="Force CPU inference (default: True)",
    )
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference",
    )
    p.add_argument(
        "--skip-punctuation",
        action="store_true",
        help="Skip the Cadence step (use if text already has dandas)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output .jsonl path (default: sanskrit_chunks_<strategy>.jsonl)",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Randomly sample N segments from the input (default: use all)",
    )
    p.add_argument(
        "--min-length",
        type=int,
        default=30,
        help="Skip segments shorter than this (default: 30 chars)",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu:
        args.cpu = False

    if args.output is None:
        args.output = f"sanskrit_chunks_{args.strategy}.jsonl"

    # ---- Demo mode ----
    if args.demo:
        run_demo(args)
        return

    # ---- Load segments ----
    if args.segments_dir:
        segments = load_segments_from_dir(args.segments_dir, args.min_length)
    elif args.input_file:
        segments = load_segments_from_txt(args.input_file, args.min_length)
    elif args.jsonl_input:
        segments = load_segments_from_jsonl(args.jsonl_input, args.min_length)
    else:
        print("Error: supply --segments-dir, --input-file, --jsonl-input, or --demo", file=sys.stderr)
        sys.exit(1)

    if not segments:
        print("No segments loaded -- check your input path.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(segments)} segments")

    # ---- Optional sampling ----
    if args.sample_size and args.sample_size < len(segments):
        rng = random.Random(args.seed)
        segments = rng.sample(segments, args.sample_size)
        print(f"Sampled {len(segments)} segments")

    # ---- Run pipeline ----
    rows = run_pipeline(
        segments,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model_name=args.model,
        batch_size=args.batch_size,
        cpu=args.cpu,
        skip_punctuation=args.skip_punctuation,
    )

    # ---- Show & save ----
    print()
    print_samples(rows)
    save_for_eval(rows, args.output, args.strategy, args.chunk_size)


if __name__ == "__main__":
    main()
