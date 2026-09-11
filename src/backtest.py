"""Walk-forward backtest: replay historical matches in date order, predict
each one with the model *before* absorbing its result into the Elo ratings,
then score the predictions once the full history has been played back.

Accuracy = share of decisive (non-draw) matches where the model's implied
favourite actually won.
Brier score = mean squared error between the predicted 3-way probability
vector [home_win, draw, away_win] and the one-hot actual outcome, averaged
over all matches (lower is better; 0 is a perfect forecast).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from dixon_coles import match_probabilities
from elo import EloRatingSystem


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
        home_adv = 0.0 if m.neutral else 100.0
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


if __name__ == "__main__":
    import os

    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "..", "data", "sample_historical_matches.csv")
    elo = EloRatingSystem()
    matches = load_matches(csv_path)
    results = walk_forward_backtest(matches, elo)
    print(f"Matches scored:        {results['matches_scored']}")
    print(f"Decisive matches:      {results['decisive_matches']}")
    print(f"Accuracy (decisive):   {results['accuracy_decisive']:.3f}")
    print(f"Brier score:           {results['brier_score']:.3f}")
