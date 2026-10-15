from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETO_COMMIT = "a2fc5da38f8d00cfaf3f9b6370d586eebaf72904"
WEBSHOP_COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
DEFAULT_DATASET = ROOT / "data/datasets/webshop/teacher_train_2000_seed42/tasks.jsonl"
DEFAULT_ETO = ROOT / "_local/upstream/ETO"


def canonical(text: str) -> str:
    # Official human_goals.json removes punctuation and price suffixes.
    text = text.split(", and price lower than", 1)[0]
    return re.sub(r"[^a-z0-9]", "", text.lower())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def observation_message(observation, clickables, done=False):
    clicks = list(dict.fromkeys(str(x).lower() for x in clickables if str(x).lower() != "search"))
    return (f"Observation:\n{observation}\n\nAvailable clicks: "
            + json.dumps([] if done else clicks, ensure_ascii=False)
            + ("\nSearch available: no (episode ended)." if done else "\nSearch available: yes (non-empty keywords)."))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    temp.replace(path)


def validate_tasks(rows):
    if not rows:
        raise ValueError("Empty task dataset")
    ids = [r["task_id"] for r in rows]
    if any(type(i) is not int or not 1500 <= i < 12087 for i in ids):
        raise ValueError("Only official training goal indices 1500..12086 are allowed")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    keys = [canonical(r["instruction_reference"]) for r in rows]
    if any(not k for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicate instructions")
    for r in rows:
        if r["split"] != "train" or canonical(r["instruction"]) != canonical(r["instruction_reference"]):
            raise ValueError("Invalid split or instruction identity")
