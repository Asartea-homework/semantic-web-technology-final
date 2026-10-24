# Running

## Setup

Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv install
```

Run the data cleaning script to generate the `stories.jsonl` file:

```bash
uv run data_cleaner.py
```

## Embeddings
The semantic recommender needs precomputed embeddings (not stored in git).
Download `embeddings.zip` from the latest release (Releases tab) and unzip
it in the repo root, so you have `embeddings/embeddings.npy`.