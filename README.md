# DharmaNexus Evaluation

A retrieval evaluation benchmark for Sanskrit texts, with a chunking pipeline that uses NLP-based punctuation restoration to split text at natural boundaries.

## What This Project Does

Sanskrit texts from digital corpora often lack punctuation, which makes it hard to split them into meaningful pieces for search and retrieval. This project solves that with a three-step pipeline:

1. **Normalize** the raw text (fix encoding, anusvara, visarga)
2. **Add punctuation** using a neural model (restore dandas where they belong)
3. **Chunk** the text into retrieval-ready segments using different splitting strategies

You can then **benchmark** which chunking strategy works best by feeding the chunks into the Dharmamitra search API and measuring how often each chunk retrieves its source passage.

## How It Works

```
                           sanskrit_pipeline.py
 ┌──────────────┐    ┌─────────────────────────────────────────────┐    ┌──────────────┐
 │  Sanskrit     │    │                                             │    │  Chunks      │
 │  Corpus       │───>│  Normalize ──> Punctuate ──> Chunk          │───>│  (.jsonl)    │
 │  (.json/.txt) │    │  (Vedika)      (Cadence)     (4 strategies) │    │              │
 └──────────────┘    └─────────────────────────────────────────────┘    └──────┬───────┘
                                                                              │
                                                                              v
                                                                    ┌──────────────────┐
                                                                    │   run_eval.py     │
                                                                    │  Query Dharmamitra │
                                                                    │  API, measure     │
                                                                    │  Recall@K         │
                                                                    └──────────────────┘
```

Each input passage gets split into one or more chunks. Each chunk becomes a row in the output file. When you run evaluation, the API is queried with each chunk, and success means the API found the original source passage. **Higher Recall@K = better chunking strategy** -- the chunks preserve enough meaning to find the source.

## Pipeline Steps

### Step 1: Normalize (Vedika)

Cleans up character-level encoding issues using the [Vedika](https://github.com/tanuj437/Vedika) toolkit:
- Unicode NFC normalization
- Anusvara correction (nasal consonant based on following letter)
- Visarga standardization
- Whitespace and control character cleanup

### Step 2: Punctuate (Cadence)

Restores missing sentence-boundary punctuation using [Cadence-Fast](https://huggingface.co/ai4bharat/Cadence-Fast), a multilingual punctuation restoration model by AI4Bharat. It predicts where dandas, double-dandas, commas, and other marks belong. This is what makes sentence-level chunking possible on texts that originally had no punctuation.

Can be skipped with `--skip-punctuation` if the input text already has dandas.

### Step 3: Chunk (4 strategies)

Splits the punctuated text into pieces. Each strategy produces a different set of chunks from the same input.

## Chunking Strategies

### sentence

Splits at danda / double-danda boundaries. Each sentence becomes its own chunk.

```
Input:  "धर्मो रक्षति रक्षितः। सत्यं वद। धर्मं चर।"
Output: ["धर्मो रक्षति रक्षितः", "सत्यं वद", "धर्मं चर"]
```

### fixed_size

Groups sentences into chunks of approximately N words (set by `--chunk-size`). Breaks at the nearest sentence boundary to avoid cutting mid-sentence.

```
--chunk-size 6
Input:  "धर्मो रक्षति रक्षितः। सत्यं वद। धर्मं चर। स्वाध्यायान्मा प्रमदः।"
Output: ["धर्मो रक्षति रक्षितः सत्यं वद", "धर्मं चर स्वाध्यायान्मा प्रमदः"]
```

### sliding_window

Overlapping word-windows of N words with configurable overlap (default 25%). Every part of the text appears in at least one chunk, with redundancy at boundaries.

```
--chunk-size 6 --overlap 0.5
Input:  "a b c d e f g h i"
Output: ["a b c d e f", "d e f g h i"]
```

### hierarchical

Produces chunks at three granularity levels from the same text:
- **Level 1**: Individual sentences
- **Level 2**: Pairs of consecutive sentences
- **Level 3**: Groups of 3 consecutive sentences

All levels go into the output, so retrieval can be tested at each granularity.

## Inputs

### DharmaNexus JSON segments (`--segments-dir`)

A directory of `.json` files where each file contains a list of segment objects:

```json
[
  { "segmentnr": "SA_GRETIL_001:42", "original": "Sanskrit text here" },
  { "segmentnr": "SA_GRETIL_001:43", "original": "More Sanskrit text" }
]
```

### Plain text file (`--input-file`)

A UTF-8 `.txt` file where each line is one passage:

```
धर्मो रक्षति रक्षितः सत्यं वद धर्मं चर
यदा यदा हि धर्मस्य ग्लानिर्भवति भारत
```

### Built-in demo (`--demo`)

Five hardcoded Sanskrit verses for quick testing. No files needed.

## Output

A `.jsonl` file (one JSON object per line) compatible with `run_eval.py`:

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

The field names `corrupted`, `corruption_level`, and `corruption_type` are kept for compatibility with `run_eval.py`. The content in `corrupted` is a clean chunk (not corrupted text). `corruption_level` stores the chunk size, and `corruption_type` stores the strategy name.

## Benchmarking

To compare chunking strategies:

```bash
# Generate chunks with each strategy
python sanskrit_pipeline.py --segments-dir ./segments --strategy sentence -o chunks_sentence.jsonl --skip-punctuation
python sanskrit_pipeline.py --segments-dir ./segments --strategy fixed_size --chunk-size 30 -o chunks_fixed.jsonl --skip-punctuation
python sanskrit_pipeline.py --segments-dir ./segments --strategy sliding_window --chunk-size 30 -o chunks_sliding.jsonl --skip-punctuation
python sanskrit_pipeline.py --segments-dir ./segments --strategy hierarchical -o chunks_hierarchical.jsonl --skip-punctuation

# Evaluate each against the Dharmamitra search API
copy chunks_sentence.jsonl eval_dataset.jsonl
python run_eval.py --languages sa --corruption-types sentence --samples-per-lang 10

copy chunks_fixed.jsonl eval_dataset.jsonl
python run_eval.py --languages sa --corruption-types fixed_size --samples-per-lang 10
```

The strategy with the highest Recall@K produces chunks that best preserve enough meaning for retrieval.

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

# Log in to HuggingFace (needed for Cadence model download)
python -c "from huggingface_hub import login; login()"
```

### Getting Sanskrit Data

```bash
git clone https://github.com/dharmamitra/dharmanexus-sanskrit
```

If not publicly accessible, the underlying texts come from [GRETIL](https://gretil.sub.uni-goettingen.de/).

## Usage

```bash
# Quick demo (no data needed, skips model download)
python sanskrit_pipeline.py --demo --skip-punctuation

# Demo with Cadence punctuation (downloads model on first run)
python sanskrit_pipeline.py --demo

# Process DharmaNexus segments
python sanskrit_pipeline.py --segments-dir /path/to/segments --strategy sentence

# Process a text file with sliding window chunks
python sanskrit_pipeline.py --input-file texts.txt --strategy sliding_window --chunk-size 40 --overlap 0.3

# Sample 100 segments and use hierarchical chunking
python sanskrit_pipeline.py --segments-dir ./segments --strategy hierarchical --sample-size 100
```

## CLI Reference

```
sanskrit_pipeline.py
  --segments-dir DIR        DharmaNexus .json segment directory
  --input-file FILE         Plain UTF-8 .txt file (one passage per line)
  --demo                    Built-in sample texts

  --strategy STRATEGY       sentence | fixed_size | sliding_window | hierarchical
  --chunk-size N            Target words per chunk (default: 50)
  --overlap FRAC            Overlap for sliding_window, 0.0-1.0 (default: 0.25)

  --model MODEL             Cadence or Cadence-Fast (default: Cadence-Fast)
  --batch-size N            Cadence batch size (default: 8)
  --cpu / --gpu             Inference device (default: cpu)
  --skip-punctuation        Skip Cadence if text already has dandas

  --output, -o FILE         Output .jsonl path
  --seed N                  Random seed (default: 42)
  --sample-size N           Sample N segments from input
  --min-length N            Skip segments shorter than N chars (default: 30)
```

## Files

| File | Purpose |
|---|---|
| `sanskrit_pipeline.py` | Chunking pipeline: normalize, punctuate, chunk, save |
| `generate_eval_dataset.py` | Original eval dataset generator (crop/mask corruption across 4 languages) |
| `run_eval.py` | Evaluation: query Dharmamitra API with chunks, measure Recall@K |
| `eval_dataset.jsonl` | Pre-generated evaluation dataset |
| `requirements.txt` | Python dependencies |

## Dependencies

| Package | Version | Role |
|---|---|---|
| [cadence-punctuation](https://pypi.org/project/cadence-punctuation/) | >= 1.1.0 | Punctuation restoration |
| [vedika](https://pypi.org/project/vedika/) | >= 0.0.18 | Sanskrit text normalization and sentence splitting |
| [torch](https://pytorch.org/) | >= 2.1.0 | Neural network runtime for Cadence |
| [transformers](https://huggingface.co/docs/transformers/) | >= 4.51.3, < 5.0.0 | Model loading |
| [huggingface_hub](https://pypi.org/project/huggingface-hub/) | >= 0.30.2 | Model downloads |
