"""World Football Elo rating system.

Implements the well-known public "World Football Elo Ratings" formula
(https://www.eloratings.net/about), which is the same rating logic used by
most football forecasting models. Ratings live in a single dict keyed by
team name so they can be updated incrementally as new results come in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HOME_ADVANTAGE = 100.0
DEFAULT_RATING = 1500.0


def expected_score(rating_a: float, rating_b: float, home_advantage: float = 0.0) -> float:
    """Win expectancy for team A given both Elo ratings."""
    diff = rating_a - rating_b + home_advantage
    return 1.0 / (10.0 ** (-diff / 400.0) + 1.0)


def goal_diff_multiplier(goal_diff: int) -> float:
    """Elo's non-linear bonus for larger margins of victory."""
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11.0 + goal_diff) / 8.0


def match_importance_k(stage: str = "friendly") -> float:
    """K-factor by competition weight, mirroring eloratings.net tiers."""
    return {
        "friendly": 20.0,
        "qualifier": 40.0,
        "continental": 50.0,
        "world_cup_group": 55.0,
        "world_cup_knockout": 60.0,
    }.get(stage, 30.0)


@dataclass
class EloRatingSystem:
    ratings: dict[str, float] = field(default_factory=dict)
    default_rating: float = DEFAULT_RATING

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def set(self, team: str, rating: float) -> None:
        self.ratings[team] = rating

    def update_match(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        stage: str = "friendly",
        neutral_site: bool = False,
    ) -> tuple[float, float]:
        home_rating = self.get(home)
        away_rating = self.get(away)
        home_adv = 0.0 if neutral_site else HOME_ADVANTAGE

        expected_home = expected_score(home_rating, away_rating, home_adv)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        k = match_importance_k(stage)
        g = goal_diff_multiplier(home_goals - away_goals)
        delta = k * g * (actual_home - expected_home)

        new_home = home_rating + delta
        new_away = away_rating - delta
        self.set(home, new_home)
        self.set(away, new_away)
        return new_home, new_away
