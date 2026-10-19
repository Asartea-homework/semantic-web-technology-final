import json
from pathlib import Path

REQUIRED_FIELDS = ["id", "title", "main_text", "tag"]
OPTIONAL_FIELDS = [
    "identifier",
    "language",
    "subgenre",
    "motif",
    "description",
    "date_start",
    "date_end",
    "decennium_group",
    "item_type",
    "title",
]


def transform_doc(doc):
    new_doc = {}
    for field in REQUIRED_FIELDS:
        if field not in doc:
            raise ValueError(f"Missing required field: {field}")
        new_doc[field] = doc[field]
    for field in OPTIONAL_FIELDS:
        if field in doc:
            new_doc[field] = doc[field]
        else:
            new_doc[field] = None
    return new_doc


def read_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    docs = data["response"]["docs"]
    output_docs = []
    err_count = 0
    for doc in docs:
        try:
            new_doc = transform_doc(doc)
            output_docs.append(new_doc)
        except ValueError as _:
            err_count += 1
            continue
    print(
        f"Processed {len(output_docs)} documents from {file_path}, {err_count} errors."
    )
    return output_docs


def main():
    data_dir = Path("data/")
    output_dir = Path("output/")
    docs = []
    files = data_dir.glob("*.json")
    for file in files:
        docs.extend(read_file(file))
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "output.jsonl", "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
