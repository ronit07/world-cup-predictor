"""Turn the real international results dataset into the CSV format the
backtest already uses.

The raw data is martj42/international_results (CC0), which has every men's
international since 1872. I copied results.csv into data/raw/ so the repo
works offline. This script maps each tournament name to one of the stages
the Elo code knows about and writes data/real_matches.csv.

Usage:
    python src/real_data.py
"""

from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "..", "data", "raw", "results.csv")
OUT_PATH = os.path.join(HERE, "..", "data", "real_matches.csv")

# big continental tournaments (finals, not qualifying)
CONTINENTAL = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
    "Confederations Cup",
    "UEFA Nations League",
    "CONCACAF Nations League",
}


def stage_for(tournament: str) -> str:
    t = tournament.strip()
    if t == "Friendly":
        return "friendly"
    if t == "FIFA World Cup":
        # the raw data doesn't say group vs knockout, so I treat all WC games
        # as group weight. Knockout K is only 5 higher so it barely matters.
        return "world_cup_group"
    if "qualification" in t:
        return "qualifier"
    if t in CONTINENTAL:
        return "continental"
    # everything else (regional cups, minor tournaments)
    return "other"


def convert(raw_path: str = RAW_PATH, out_path: str = OUT_PATH) -> int:
    rows_out = 0
    with open(raw_path, newline="", encoding="utf-8") as fin, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(
            ["date", "home_team", "away_team", "home_goals", "away_goals", "stage", "neutral", "country"]
        )
        for row in reader:
            # a few future fixtures have NA scores, skip them
            if row["home_score"] in ("NA", "") or row["away_score"] in ("NA", ""):
                continue
            writer.writerow(
                [
                    row["date"],
                    row["home_team"],
                    row["away_team"],
                    int(row["home_score"]),
                    int(row["away_score"]),
                    stage_for(row["tournament"]),
                    row["neutral"].strip().upper() == "TRUE",
                    row["country"],
                ]
            )
            rows_out += 1
    return rows_out


if __name__ == "__main__":
    n = convert()
    print(f"wrote {n} matches to {os.path.relpath(OUT_PATH)}")
