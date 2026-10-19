# Running

## Setup

Create a virtual environment and install dependencies:

```bash
uv venv
source venv/bin/activate
uv install
```

Run the data cleaning script to generate the `stories.jsonl` file:

```bash
uv run data_cleaner.py
```
