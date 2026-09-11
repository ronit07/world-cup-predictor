"""Generates a synthetic match-history CSV for backtesting demos.

This is NOT a claim of real historical scorelines -- it draws matches from
the same Dixon-Coles model using the seed ratings in data/seed_ratings.csv,
so the backtest has a self-consistent dataset to walk forward through. Swap
this file for a real results dataset (e.g. the public "international
football results" archive) to backtest against actual history.
"""

from __future__ import annotations

import csv
import os
import random

from dixon_coles import match_probabilities
from simulate import sample_scoreline


def load_seed_ratings(path: str) -> dict[str, float]:
    ratings = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ratings[row["team"]] = float(row["seed_elo"])
    return ratings


def generate(
    ratings: dict[str, float],
    n_matches: int,
    seed: int = 7,
    start_year: int = 2018,
    end_year: int = 2025,
) -> list[dict]:
    rng = random.Random(seed)
    teams = list(ratings.keys())
    rows = []
    for _ in range(n_matches):
        home, away = rng.sample(teams, 2)
        neutral = rng.random() < 0.3
        home_adv = 0.0 if neutral else 100.0
        probs = match_probabilities(ratings[home], ratings[away], home_advantage=home_adv)
        hg, ag = sample_scoreline(probs["scoreline_matrix"], rng)
        year = rng.randint(start_year, end_year)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        stage = rng.choice(["friendly", "qualifier", "continental"])
        rows.append(
            {
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "home_team": home,
                "away_team": away,
                "home_goals": hg,
                "away_goals": ag,
                "stage": stage,
                "neutral": str(neutral),
            }
        )
    rows.sort(key=lambda r: r["date"])
    return rows


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    ratings = load_seed_ratings(os.path.join(here, "..", "data", "seed_ratings.csv"))
    rows = generate(ratings, n_matches=600)
    out_path = os.path.join(here, "..", "data", "sample_historical_matches.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "home_team", "away_team", "home_goals", "away_goals", "stage", "neutral"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic matches to {out_path}")
