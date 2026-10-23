import json
from pathlib import Path


def load_stories(data_path: Path) -> dict[str, dict]:
    stories = {}

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            story = json.loads(line)
            stories[story["id"]] = story

    return stories
