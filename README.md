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

To test if it works: 
    - in main.py change this line: 

```python
    recommender = MetadataRecommender(stories)
```

    To:
```python
    recommender = SemanticRecommender(stories)
```

    - Then run it from the terminal:
```bash
    uv sync
    uv run main.py
```