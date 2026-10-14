"""Shared helpers for locating and loading the project's data files.

Paths are resolved relative to the repository, so every script works no
matter which folder you run it from.
"""

from __future__ import annotations

import csv
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
SEED_RATINGS_PATH = os.path.join(DATA_DIR, "seed_ratings.csv")
GROUPS_PATH = os.path.join(DATA_DIR, "groups.json")
SAMPLE_MATCHES_PATH = os.path.join(DATA_DIR, "sample_historical_matches.csv")


def load_seed_ratings(path: str = SEED_RATINGS_PATH) -> dict[str, float]:
    """Read `team,seed_elo` rows into a {team: rating} dict."""
    ratings = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ratings[row["team"]] = float(row["seed_elo"])
    return ratings


def load_groups(path: str = GROUPS_PATH) -> dict[str, list[str]]:
    """Read the group draw, skipping comment keys such as `_comment`."""
    with open(path, encoding="utf-8") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
