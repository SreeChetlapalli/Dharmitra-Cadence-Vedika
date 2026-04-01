# DharmaNexus Evaluation

Benchmark different ways of splitting Sanskrit text for search and retrieval.

## Quick Start

Everything below assumes you have Python 3.9+ and have already set up the virtual environment. If not, jump to [Setup](#setup) first.

### 1. Extract Sanskrit passages from the pre-generated dataset

The repo ships with `eval_dataset.jsonl`, which contains 1,000 Sanskrit passages (plus Tibetan, Chinese, and Pali). Extract just the Sanskrit into a `.jsonl` file that preserves the segment IDs needed for benchmarking:

```bash
python extract_sanskrit.py
```

This creates `sanskrit_input.jsonl` -- one passage per line, each with its `segmentnr` so that benchmarking can match API results back to the source.

> **Why `.jsonl` and not `.txt`?** The `run_eval.py` benchmark checks whether the Dharmamitra API returns a result whose segment ID matches the chunk's segment ID. If you use `--input-file` with a plain `.txt`, the pipeline assigns synthetic IDs like `sanskrit_texts:1` which won't match the API, so Recall will always be 0%. Using `--jsonl-input` preserves the real segment IDs.

### 2. Run the chunking pipeline

```bash
# Quick test on 10 passages, skip Cadence model (fast)
python sanskrit_pipeline.py --jsonl-input sanskrit_input.jsonl --strategy sentence --sample-size 10 --skip-punctuation

# Full run on all 1,000 passages with Cadence punctuation
python sanskrit_pipeline.py --jsonl-input sanskrit_input.jsonl --strategy sentence
```

This produces `sanskrit_chunks_sentence.jsonl`.

### 3. Benchmark against the Dharmamitra search API

`run_eval.py` always reads from `eval_dataset.jsonl`, so copy your chunks file there first:

```bash
# Back up the original
copy eval_dataset.jsonl eval_dataset_backup.jsonl

# Use your chunks as the evaluation dataset
copy sanskrit_chunks_sentence.jsonl eval_dataset.jsonl

# Run evaluation (queries the Dharmamitra API)
python run_eval.py --languages sa --corruption-types sentence --samples-per-lang 50
```

The output shows Recall@K -- how often a chunk retrieves its source passage. Higher is better.

### 4. Compare strategies

Repeat steps 2-3 with different strategies:

```bash
python sanskrit_pipeline.py --jsonl-input sanskrit_input.jsonl --strategy fixed_size --chunk-size 30
copy sanskrit_chunks_fixed_size.jsonl eval_dataset.jsonl
python run_eval.py --languages sa --corruption-types fixed_size --samples-per-lang 50

python sanskrit_pipeline.py --jsonl-input sanskrit_input.jsonl --strategy sliding_window --chunk-size 30 --overlap 0.25
copy sanskrit_chunks_sliding_window.jsonl eval_dataset.jsonl
python run_eval.py --languages sa --corruption-types sliding_window --samples-per-lang 50

python sanskrit_pipeline.py --jsonl-input sanskrit_input.jsonl --strategy hierarchical
copy sanskrit_chunks_hierarchical.jsonl eval_dataset.jsonl
python run_eval.py --languages sa --corruption-types hierarchical --samples-per-lang 50
```

The strategy with the highest Recall@K is the best chunking method for search.

---

## What This Project Does

Sanskrit texts from digital corpora often lack punctuation, which makes it hard to split them into meaningful pieces for search and retrieval. This project solves that with a three-step pipeline:

1. **Normalize** the raw text (fix encoding, anusvara, visarga)
2. **Add punctuation** using a neural model (restore dandas where they belong)
3. **Chunk** the text into retrieval-ready segments using different splitting strategies

You then benchmark which chunking strategy works best by feeding the chunks into the Dharmamitra search API and measuring Recall@K.

### How it flows

```
                           sanskrit_pipeline.py
 ┌──────────────┐    ┌─────────────────────────────────────────────┐    ┌──────────────┐
 │  Sanskrit     │    │                                             │    │  Chunks      │
 │  text         │───>│  Normalize ──> Punctuate ──> Chunk          │───>│  (.jsonl)    │
 │  (.jsonl/     │    │  (Vedika)      (Cadence)     (4 strategies) │    │              │
 │   .txt/.json) │    │                                             │    │              │
 └──────────────┘    └─────────────────────────────────────────────┘    └──────┬───────┘
                                                                              │
                                                                         copy to
                                                                      eval_dataset.jsonl
                                                                              │
                                                                              v
                                                                    ┌──────────────────┐
                                                                    │   run_eval.py     │
                                                                    │  Query Dharmamitra │
                                                                    │  API, measure     │
                                                                    │  Recall@K         │
                                                                    └──────────────────┘
```

---

## Pipeline Steps

### Step 1: Normalize (Vedika)

Cleans up character-level encoding issues using [Vedika](https://github.com/tanuj437/Vedika):
- Unicode NFC normalization
- Anusvara correction (nasal consonant based on following letter)
- Visarga standardization
- Whitespace and control character cleanup

### Step 2: Punctuate (Cadence)

Restores missing sentence-boundary punctuation using [Cadence](https://huggingface.co/ai4bharat/Cadence-Fast), a multilingual model by AI4Bharat. It predicts where dandas (`।`), double-dandas (`॥`), commas, and other marks belong. This is what makes sentence-level chunking possible on texts that originally had no punctuation.

Skip this step with `--skip-punctuation` if your text already has dandas.

Choose the model variant with `--model`:
- `Cadence-Fast` (default) -- smaller, ~4x faster, 93.8% of full quality
- `Cadence` -- full 1B parameter model, highest accuracy

### Step 3: Chunk (4 strategies)

Splits the punctuated text into pieces. No text is removed or corrupted -- every word is preserved.

---

## Chunking Strategies

### `sentence`

Splits at danda / double-danda boundaries. Each sentence becomes its own chunk.

```
Input:  "धर्मो रक्षति रक्षितः। सत्यं वद। धर्मं चर।"
Output: ["धर्मो रक्षति रक्षितः", "सत्यं वद", "धर्मं चर"]
```

### `fixed_size`

Groups sentences into chunks of approximately N words (set by `--chunk-size`). Breaks at the nearest sentence boundary to avoid cutting mid-sentence.

```
--chunk-size 6
Input:  "धर्मो रक्षति रक्षितः। सत्यं वद। धर्मं चर। स्वाध्यायान्मा प्रमदः।"
Output: ["धर्मो रक्षति रक्षितः सत्यं वद", "धर्मं चर स्वाध्यायान्मा प्रमदः"]
```

### `sliding_window`

Overlapping word-windows of N words with configurable overlap (default 25%). Every part of the text appears in at least one chunk, with redundancy at boundaries.

```
--chunk-size 6 --overlap 0.5
Input:  "a b c d e f g h i"
Output: ["a b c d e f", "d e f g h i"]
```

### `hierarchical`

Produces chunks at three granularity levels from the same text:
- **Level 1**: Individual sentences
- **Level 2**: Pairs of consecutive sentences
- **Level 3**: Groups of 3 consecutive sentences

All levels go into the output, so retrieval can be tested at each granularity.

---

## Input Sources

You need one of these three options:

### Option A: Plain text file (`--input-file`)

A UTF-8 `.txt` file where each line is one passage. This is the simplest option.

```
धर्मो रक्षति रक्षितः सत्यं वद धर्मं चर
यदा यदा हि धर्मस्य ग्लानिर्भवति भारत
```

You can write your own `.txt` file for quick experiments. For benchmarking, use `--jsonl-input` instead (see below) so that segment IDs are preserved.

### Option B: JSONL file with segment IDs (`--jsonl-input`)

A `.jsonl` file where each line is a JSON object with `segmentnr` and `original` fields. This is the recommended option for benchmarking because it preserves the real segment IDs that `run_eval.py` needs.

```json
{"segmentnr": "SA_GRETIL_001:42", "original": "Sanskrit text here"}
{"segmentnr": "SA_GRETIL_001:43", "original": "More Sanskrit text"}
```

Create this by extracting from `eval_dataset.jsonl` (see [Quick Start](#quick-start)).

### Option C: DharmaNexus JSON segments (`--segments-dir`)

A directory of `.json` files where each file contains a list of segment objects:

```json
[
  { "segmentnr": "SA_GRETIL_001:42", "original": "Sanskrit text here" },
  { "segmentnr": "SA_GRETIL_001:43", "original": "More Sanskrit text" }
]
```

Get these by cloning `https://github.com/dharmamitra/dharmanexus-sanskrit` (if publicly accessible) or from [GRETIL](https://gretil.sub.uni-goettingen.de/).

### Option D: Built-in demo (`--demo`)

Five hardcoded Sanskrit verses for quick testing. No files needed.

```bash
python sanskrit_pipeline.py --demo --skip-punctuation
```

---

## Output Format

The pipeline produces a `.jsonl` file (one JSON object per line) compatible with `run_eval.py`:

```json
{
  "language": "sa",
  "segmentnr": "SA_GRETIL_001:42",
  "original": "full passage text",
  "corrupted": "one chunk of that passage",
  "corruption_level": 50,
  "corruption_type": "sentence"
}
```

The field names `corrupted`, `corruption_level`, and `corruption_type` exist for compatibility with `run_eval.py`. Despite the names:
- `corrupted` contains a **clean chunk** (nothing is corrupted)
- `corruption_level` stores the **word count** of the chunk
- `corruption_type` stores the **strategy name**

---

## Setup

### Prerequisites

- Python 3.9+
- A HuggingFace account (for Cadence model weights)

### Installation

```bash
git clone https://github.com/dharmamitra/dharmanexus-evaluation
cd dharmanexus-evaluation

python -m venv mitraVenv
# Windows:
mitraVenv\Scripts\activate
# Linux/Mac:
source mitraVenv/bin/activate

pip install -r requirements.txt
```

### HuggingFace login (needed for Cadence)

```bash
python -c "from huggingface_hub import login; login()"
```

If that fails due to network issues, save your token manually:

```bash
python -c "from huggingface_hub import HfFolder; HfFolder.save_token('YOUR_TOKEN_HERE')"
```

Get your token from https://huggingface.co/settings/tokens.

---

## CLI Reference

```
sanskrit_pipeline.py

Input (pick one):
  --jsonl-input FILE      .jsonl file with segmentnr + original (best for benchmarking)
  --input-file FILE       Plain UTF-8 .txt file (one passage per line)
  --segments-dir DIR      DharmaNexus .json segment directory
  --demo                  Run on built-in sample texts

Chunking:
  --strategy STRATEGY     sentence | fixed_size | sliding_window | hierarchical (default: sentence)
  --chunk-size N          Target words per chunk (default: 50)
  --overlap FRAC          Overlap fraction for sliding_window, 0.0-1.0 (default: 0.25)

Cadence model:
  --model MODEL           Cadence or Cadence-Fast (default: Cadence-Fast)
  --batch-size N          Cadence batch size (default: 8)
  --cpu                   Force CPU inference (default)
  --gpu                   Use GPU for inference
  --skip-punctuation      Skip Cadence entirely (use if text already has dandas)

Output:
  --output, -o FILE       Output .jsonl path (default: sanskrit_chunks_<strategy>.jsonl)
  --sample-size N         Only process N randomly-selected passages (good for quick tests)
  --min-length N          Skip passages shorter than N characters (default: 30)
  --seed N                Random seed for sampling (default: 42)
```

---

## Files

| File | Purpose |
|---|---|
| `extract_sanskrit.py` | Extract Sanskrit passages from `eval_dataset.jsonl`, preserving segment IDs |
| `sanskrit_pipeline.py` | Chunking pipeline: normalize, punctuate, chunk, save |
| `run_eval.py` | Benchmark: query Dharmamitra API with chunks, measure Recall@K |
| `generate_eval_dataset.py` | Original eval dataset generator (crop/mask corruption across 4 languages) |
| `eval_dataset.jsonl` | Pre-generated dataset with 1,000 passages each in Sanskrit, Tibetan, Chinese, and Pali |
| `sanskrit_input.jsonl` | Extracted Sanskrit passages with segment IDs (created by `extract_sanskrit.py`) |
| `requirements.txt` | Python dependencies |

## Dependencies

| Package | Version | Role |
|---|---|---|
| [cadence-punctuation](https://pypi.org/project/cadence-punctuation/) | >= 1.1.0 | Punctuation restoration |
| [vedika](https://pypi.org/project/vedika/) | >= 0.0.18 | Sanskrit normalization and sentence splitting |
| [torch](https://pytorch.org/) | >= 2.1.0 | Neural network runtime for Cadence |
| [transformers](https://huggingface.co/docs/transformers/) | >= 4.51.3, < 5.0.0 | Model loading |
| [huggingface_hub](https://pypi.org/project/huggingface-hub/) | >= 0.30.2 | Model downloads |
