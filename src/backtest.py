"""Walk-forward backtest: replay historical matches in date order, predict
each one with the model *before* absorbing its result into the Elo ratings,
then score the predictions once the full history has been played back.

Accuracy = share of decisive (non-draw) matches where the model's implied
favourite actually won.
Brier score = mean squared error between the predicted 3-way probability
vector [home_win, draw, away_win] and the one-hot actual outcome, averaged
over all matches (lower is better; 0 is a perfect forecast, and always
guessing 1/3 for each outcome scores 0.667).

By default the ratings start from data/seed_ratings.csv, the same ratings
the tournament simulation uses. Pass --cold-start to begin every team at
1500 instead and watch the model learn the ratings from the results alone.

Usage:
    python src/backtest.py
    python src/backtest.py --cold-start
    python src/backtest.py --data path/to/real_results.csv
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

from data_loading import SAMPLE_MATCHES_PATH, SEED_RATINGS_PATH, load_seed_ratings
from dixon_coles import match_probabilities
from elo import HOME_ADVANTAGE, EloRatingSystem


@dataclass
class MatchRecord:
    date: str
    home: str
    away: str
    home_goals: int
    away_goals: int
    stage: str
    neutral: bool


def load_matches(csv_path: str) -> list[MatchRecord]:
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append(
                MatchRecord(
                    date=row["date"],
                    home=row["home_team"],
                    away=row["away_team"],
                    home_goals=int(row["home_goals"]),
                    away_goals=int(row["away_goals"]),
                    stage=row["stage"],
                    neutral=row["neutral"].strip().lower() == "true",
                )
            )
    records.sort(key=lambda r: r.date)
    return records


def outcome_vector(home_goals: int, away_goals: int) -> tuple[float, float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0, 0.0
    if home_goals == away_goals:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def walk_forward_backtest(matches: list[MatchRecord], elo: EloRatingSystem) -> dict:
    n_total = 0
    n_decisive = 0
    n_correct = 0
    brier_sum = 0.0

    for m in matches:
        home_adv = 0.0 if m.neutral else HOME_ADVANTAGE
        probs = match_probabilities(elo.get(m.home), elo.get(m.away), home_advantage=home_adv)
        predicted = (probs["home_win"], probs["draw"], probs["away_win"])
        actual = outcome_vector(m.home_goals, m.away_goals)

        brier_sum += sum((p - o) ** 2 for p, o in zip(predicted, actual))
        n_total += 1

        if m.home_goals != m.away_goals:
            n_decisive += 1
            favourite_home = probs["home_win"] >= probs["away_win"]
            home_won = m.home_goals > m.away_goals
            if favourite_home == home_won:
                n_correct += 1

        elo.update_match(m.home, m.away, m.home_goals, m.away_goals, stage=m.stage, neutral_site=m.neutral)

    return {
        "matches_scored": n_total,
        "decisive_matches": n_decisive,
        "accuracy_decisive": n_correct / n_decisive if n_decisive else 0.0,
        "brier_score": brier_sum / n_total if n_total else 0.0,
    }


def uniform_brier(matches: list[MatchRecord]) -> float:
    """Brier score of the no-skill baseline: 1/3 home, 1/3 draw, 1/3 away."""
    if not matches:
        return 0.0
    guess = (1 / 3, 1 / 3, 1 / 3)
    total = sum(
        sum((p - o) ** 2 for p, o in zip(guess, outcome_vector(m.home_goals, m.away_goals)))
        for m in matches
    )
    return total / len(matches)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward backtest: predict each past match before learning from it."
    )
    parser.add_argument(
        "--data",
        default=SAMPLE_MATCHES_PATH,
        help="results CSV with columns date,home_team,away_team,home_goals,away_goals,stage,neutral "
        "(default: the bundled synthetic sample)",
    )
    parser.add_argument(
        "--ratings",
        default=SEED_RATINGS_PATH,
        help="starting Elo ratings CSV (default: data/seed_ratings.csv)",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="ignore --ratings and start every team at 1500",
    )
    args = parser.parse_args()

    matches = load_matches(args.data)
    if args.cold_start:
        elo = EloRatingSystem()
        start = "every team at 1500 (cold start)"
    else:
        elo = EloRatingSystem(ratings=load_seed_ratings(args.ratings))
        start = f"seed ratings from {os.path.relpath(args.ratings)}"

    results = walk_forward_backtest(matches, elo)
    baseline = uniform_brier(matches)

    print(f"Backtest on {os.path.relpath(args.data)}")
    print(f"Starting ratings:      {start}")
    print()
    print(f"Matches scored:        {results['matches_scored']}")
    print(f"Decisive matches:      {results['decisive_matches']}  (draws are excluded from accuracy)")
    print(f"Accuracy (decisive):   {results['accuracy_decisive']:.1%}  (coin flip = 50.0%)")
    print(f"Brier score:           {results['brier_score']:.3f}  (lower is better; guessing 1/3 each = {baseline:.3f})")
    if baseline > 0:
        skill = 1 - results["brier_score"] / baseline
        print(f"Skill vs. guessing:    {skill:+.1%}  (how much the Brier score improves on the no-skill guess)")


if __name__ == "__main__":
    main()
