"""CLI entry point: run the Monte Carlo tournament simulation and print
title odds for every team, sorted highest to lowest.

Usage: python run_simulation.py [--sims 10000]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from elo import EloRatingSystem  # noqa: E402
from simulate import run_monte_carlo  # noqa: E402


def load_seed_ratings(path: str) -> dict[str, float]:
    ratings = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ratings[row["team"]] = float(row["seed_elo"])
    return ratings


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo World Cup simulator")
    parser.add_argument("--sims", type=int, default=10_000, help="number of tournament simulations")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    ratings = load_seed_ratings(os.path.join(here, "data", "seed_ratings.csv"))
    with open(os.path.join(here, "data", "groups.json"), encoding="utf-8") as f:
        groups = {k: v for k, v in json.load(f).items() if not k.startswith("_")}

    elo = EloRatingSystem(ratings=ratings)
    odds = run_monte_carlo(groups, elo, n_sims=args.sims, seed=args.seed)

    ranked = sorted(odds.items(), key=lambda kv: kv[1]["champion"], reverse=True)
    print(f"{'Team':<15}{'Champion':>10}{'Semis':>10}{'QF':>10}{'R16':>10}{'R32':>10}")
    for team, stage_odds in ranked[:20]:
        print(
            f"{team:<15}"
            f"{stage_odds['champion'] * 100:>9.1f}%"
            f"{stage_odds['semifinal'] * 100:>9.1f}%"
            f"{stage_odds['quarterfinal'] * 100:>9.1f}%"
            f"{stage_odds['round_of_16'] * 100:>9.1f}%"
            f"{stage_odds['round_of_32'] * 100:>9.1f}%"
        )


if __name__ == "__main__":
    main()
