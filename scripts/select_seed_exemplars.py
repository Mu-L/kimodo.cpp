#!/usr/bin/env python3
"""Select auditable in-distribution prompts from BONES-SEED training timelines."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CATEGORIES = (
    ("walk", re.compile(r"\bwalk(?:s|ing|ed)?\b", re.I)),
    ("run", re.compile(r"\brun(?:s|ning)?\b", re.I)),
    ("jump", re.compile(r"\b(?:jump|leap)(?:s|ing|ed)?\b", re.I)),
    ("turn", re.compile(r"\bturn(?:s|ing|ed)?\b", re.I)),
    ("crouch", re.compile(r"\b(?:crouch|squat)(?:es|s|ing|ed)?\b", re.I)),
    ("wave", re.compile(r"\bwave(?:s|ing|ed)?\b", re.I)),
    ("kick", re.compile(r"\bkick(?:s|ing|ed)?\b", re.I)),
    ("punch", re.compile(r"\bpunch(?:es|ing|ed)?\b", re.I)),
    ("dance", re.compile(r"\bdanc(?:e|es|ing|ed)\b", re.I)),
    ("sit", re.compile(r"\b(?:sit|sits|sitting|sat)\b", re.I)),
    ("reach", re.compile(r"\breach(?:es|ing|ed)?\b", re.I)),
    ("gesture", re.compile(r"\b(?:point|clap|nod|bow)(?:s|ping|ding|ing|ed)?\b", re.I)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--timelines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-seconds", type=float, default=1.5)
    parser.add_argument("--maximum-seconds", type=float, default=8.0)
    args = parser.parse_args()
    split_entries = [line.strip() for line in args.train_split.read_text().splitlines() if line.strip()]
    by_filename = {Path(entry).name: entry for entry in split_entries}
    selected: dict[str, dict[str, object]] = {}
    seen_prompts: set[str] = set()
    with args.timelines.open() as stream:
        for line in stream:
            timeline = json.loads(line)
            filename = timeline.get("filename")
            if filename not in by_filename or str(filename).endswith("_M"):
                continue
            for event in timeline.get("events", []):
                prompt = " ".join(str(event.get("description", "")).split())
                duration = float(event.get("end_time", 0)) - float(event.get("start_time", 0))
                if prompt in seen_prompts or not (args.minimum_seconds <= duration <= args.maximum_seconds):
                    continue
                for category, pattern in CATEGORIES:
                    if category not in selected and pattern.search(prompt):
                        selected[category] = {
                            "category": category,
                            "prompt": prompt,
                            "filename": filename,
                            "train_split_entry": by_filename[filename],
                            "start_time": event["start_time"],
                            "end_time": event["end_time"],
                            "duration_seconds": duration,
                        }
                        seen_prompts.add(prompt)
                        break
            if len(selected) == len(CATEGORIES):
                break
    missing = [category for category, _ in CATEGORIES if category not in selected]
    if missing:
        raise SystemExit("could not find training exemplars for: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    exemplars = []
    for index, (category, _) in enumerate(CATEGORIES, 1):
        item = selected[category]
        prompt_file = f"{index:02d}-{category}.txt"
        (args.output / prompt_file).write_text(str(item["prompt"]) + "\n")
        item["prompt_file"] = prompt_file
        exemplars.append(item)
    manifest = {
        "schema_version": 1,
        "provenance": "NVIDIA BONES-SEED timeline annotation whose filename is present in train_split_paths.txt",
        "train_split_sha256": sha256(args.train_split),
        "timelines_sha256": sha256(args.timelines),
        "exemplars": exemplars,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
