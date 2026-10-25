# Setup

## Virtual Environment

Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

## Stories

```bash
uv run data_cleaner.py
```

Run the data cleaning script to generate the `stories.jsonl` file:

## Embeddings

The semantic recommender needs precomputed embeddings (not stored in git).
Download `embeddings.zip` from the latest release (Releases tab) and unzip
it in the app root, so you have `app/embeddings/embeddings.npy`.

# Running

```bash
uv run python -m app
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.
