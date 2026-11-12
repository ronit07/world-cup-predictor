"""Build one row of features per match, using only what was known *before*
kickoff. I walk through every match in date order, record the features,
and only then update Elo and the form history with the result. That way
the model can't peek at the score it's trying to predict.

Features per match:
  - Elo of each team and the difference (home advantage handled separately)
  - Dixon-Coles win/draw/loss probabilities from the Elo gap
  - rolling form over the last 5 and 10 games: points per game and goal diff
  - neutral venue flag, and whether either team is the host country
  - how important the match is (friendly, qualifier, ...)
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque

from dixon_coles import match_probabilities
from elo import HOME_ADVANTAGE, EloRatingSystem

STAGES = ["friendly", "other", "qualifier", "continental", "world_cup_group", "world_cup_knockout"]

FEATURE_NAMES = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "dc_home_win",
    "dc_draw",
    "dc_away_win",
    "home_ppg_5",
    "away_ppg_5",
    "home_gd_5",
    "away_gd_5",
    "home_ppg_10",
    "away_ppg_10",
    "home_gd_10",
    "away_gd_10",
    "home_games_played",
    "away_games_played",
    "neutral",
    "home_is_host",
    "away_is_host",
    "stage_level",
]

# label encoding: 0 = home win, 1 = draw, 2 = away win
HOME_WIN, DRAW, AWAY_WIN = 0, 1, 2


def outcome_label(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return HOME_WIN
    if home_goals == away_goals:
        return DRAW
    return AWAY_WIN


class FormTracker:
    """Keeps each team's last 10 results as (points, goal_diff)."""

    def __init__(self, window: int = 10):
        self.history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.games_played: dict[str, int] = defaultdict(int)

    def stats(self, team: str, n: int) -> tuple[float, float]:
        recent = list(self.history[team])[-n:]
        if not recent:
            # no history yet, so use neutral-ish defaults
            return 1.3, 0.0
        ppg = sum(p for p, _ in recent) / len(recent)
        gd = sum(g for _, g in recent) / len(recent)
        return ppg, gd

    def add(self, team: str, goals_for: int, goals_against: int) -> None:
        if goals_for > goals_against:
            pts = 3
        elif goals_for == goals_against:
            pts = 1
        else:
            pts = 0
        self.history[team].append((pts, goals_for - goals_against))
        self.games_played[team] += 1


def load_real_matches(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "date": r["date"],
                    "home": r["home_team"],
                    "away": r["away_team"],
                    "home_goals": int(r["home_goals"]),
                    "away_goals": int(r["away_goals"]),
                    "stage": r["stage"],
                    "neutral": r["neutral"].strip().lower() == "true",
                    "country": r.get("country", ""),
                }
            )
    rows.sort(key=lambda r: r["date"])
    return rows


def match_features(
    elo: EloRatingSystem, form: FormTracker, home: str, away: str, neutral: bool, country: str, stage: str
) -> list[float]:
    elo_h, elo_a = elo.get(home), elo.get(away)
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    dc = match_probabilities(elo_h, elo_a, home_advantage=home_adv)
    h5, a5 = form.stats(home, 5), form.stats(away, 5)
    h10, a10 = form.stats(home, 10), form.stats(away, 10)
    return [
        elo_h,
        elo_a,
        elo_h - elo_a,
        dc["home_win"],
        dc["draw"],
        dc["away_win"],
        h5[0],
        a5[0],
        h5[1],
        a5[1],
        h10[0],
        a10[0],
        h10[1],
        a10[1],
        min(form.games_played[home], 50),
        min(form.games_played[away], 50),
        float(neutral),
        float(country == home),
        float(country == away),
        float(STAGES.index(stage) if stage in STAGES else 1),
    ]


def build_dataset(matches: list[dict]):
    """Returns (rows, elo, form). Each row has date, features, label and the
    plain Elo/Dixon-Coles probabilities so I can compare against the old model."""
    elo = EloRatingSystem()
    form = FormTracker()
    rows = []
    for m in matches:
        feats = match_features(elo, form, m["home"], m["away"], m["neutral"], m["country"], m["stage"])
        rows.append(
            {
                "date": m["date"],
                "home": m["home"],
                "away": m["away"],
                "features": feats,
                "label": outcome_label(m["home_goals"], m["away_goals"]),
                "dc_probs": (feats[3], feats[4], feats[5]),
            }
        )
        # only now learn from the result
        elo.update_match(
            m["home"], m["away"], m["home_goals"], m["away_goals"], stage=m["stage"], neutral_site=m["neutral"]
        )
        form.add(m["home"], m["home_goals"], m["away_goals"])
        form.add(m["away"], m["away_goals"], m["home_goals"])
    return rows, elo, form
