"""Monte Carlo tournament simulation.

Simulates the 2026-format World Cup (12 groups of 4 -> top 2 per group plus
the 8 best third-place teams -> 32-team knockout bracket) many times over,
sampling each match's scoreline from the Dixon-Coles matrix, to produce
title / advancement odds per team.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from dixon_coles import match_probabilities
from elo import EloRatingSystem, HOME_ADVANTAGE


@dataclass
class TeamStanding:
    team: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against

    def sort_key(self) -> tuple:
        return (self.points, self.goal_diff, self.goals_for)


def sample_scoreline(matrix: list[list[float]], rng: random.Random) -> tuple[int, int]:
    r = rng.random()
    cumulative = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            cumulative += p
            if r <= cumulative:
                return i, j
    return len(matrix) - 1, len(matrix[0]) - 1


def simulate_match(
    home: str, away: str, elo: EloRatingSystem, rng: random.Random, neutral: bool = True
) -> tuple[int, int]:
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    probs = match_probabilities(elo.get(home), elo.get(away), home_advantage=home_adv)
    return sample_scoreline(probs["scoreline_matrix"], rng)


def simulate_group(
    teams: list[str], elo: EloRatingSystem, rng: random.Random
) -> list[TeamStanding]:
    standings = {t: TeamStanding(t) for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            home, away = teams[i], teams[j]
            hg, ag = simulate_match(home, away, elo, rng)
            standings[home].goals_for += hg
            standings[home].goals_against += ag
            standings[away].goals_for += ag
            standings[away].goals_against += hg
            if hg > ag:
                standings[home].points += 3
            elif hg < ag:
                standings[away].points += 3
            else:
                standings[home].points += 1
                standings[away].points += 1
    return sorted(standings.values(), key=lambda s: s.sort_key(), reverse=True)


def _knockout_winner(team_a: str, team_b: str, elo: EloRatingSystem, rng: random.Random) -> str:
    """Single-elimination match: draws go to a penalty-shootout coin flip
    weighted by the same win-probability model (extra time rarely changes
    the pre-match favourite by much, so this is a reasonable approximation)."""
    probs = match_probabilities(elo.get(team_a), elo.get(team_b), home_advantage=0.0)
    hg, ag = sample_scoreline(probs["scoreline_matrix"], rng)
    if hg != ag:
        return team_a if hg > ag else team_b
    # Draw -> shootout, weighted by regulation win-probability split of the two sides.
    p_a = probs["home_win"] / (probs["home_win"] + probs["away_win"] + 1e-9)
    return team_a if rng.random() < p_a else team_b


STAGE_LABELS = {16: "round_of_16", 8: "quarterfinal", 4: "semifinal", 2: "final", 1: "champion"}


def simulate_knockout_bracket(qualifiers: list[str], elo: EloRatingSystem, rng: random.Random) -> dict:
    """Plays single-elimination rounds until one champion remains, labeling
    each winner set by the stage it reached. `qualifiers` must be a power of
    two (already seeded/ordered into bracket pairs)."""
    if len(qualifiers) & (len(qualifiers) - 1) != 0:
        raise ValueError(f"knockout field must be a power of two, got {len(qualifiers)}")

    round_results = {}
    current_round = list(qualifiers)
    while len(current_round) > 1:
        next_round = [
            _knockout_winner(current_round[i], current_round[i + 1], elo, rng)
            for i in range(0, len(current_round), 2)
        ]
        label = STAGE_LABELS.get(len(next_round), f"round_of_{len(next_round)}")
        round_results[label] = next_round
        current_round = next_round
    return round_results


def pick_third_place_qualifiers(third_place: list[TeamStanding], n: int = 8) -> list[TeamStanding]:
    return sorted(third_place, key=lambda s: s.sort_key(), reverse=True)[:n]


def simulate_tournament(
    groups: dict[str, list[str]], elo: EloRatingSystem, rng: random.Random
) -> dict:
    """Runs one full tournament and returns how far every team advanced."""
    advancement = {}  # team -> furthest stage reached
    group_winners, group_runners_up, third_place = [], [], []

    for group_name, teams in groups.items():
        standings = simulate_group(teams, elo, rng)
        for t in teams:
            advancement[t] = "group_stage"
        group_winners.append(standings[0].team)
        group_runners_up.append(standings[1].team)
        third_place.append(standings[2])

    best_thirds = [s.team for s in pick_third_place_qualifiers(third_place)]
    r32_field = group_winners + group_runners_up + best_thirds
    for t in r32_field:
        advancement[t] = "round_of_32"

    rng.shuffle(r32_field)  # simplified seeding; a real bracket would use fixed slotting
    bracket = simulate_knockout_bracket(r32_field, elo, rng)

    for label, teams in bracket.items():
        for t in teams:
            advancement[t] = label

    return {"champion": bracket["champion"][0], "advancement": advancement}


STAGE_ORDER = [
    "group_stage",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
]


def run_monte_carlo(
    groups: dict[str, list[str]],
    elo: EloRatingSystem,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    champions: dict[str, int] = defaultdict(int)

    for _ in range(n_sims):
        result = simulate_tournament(groups, elo, rng)
        champions[result["champion"]] += 1
        for team, stage in result["advancement"].items():
            stage_idx = STAGE_ORDER.index(stage)
            for reached in STAGE_ORDER[: stage_idx + 1]:
                counts[team][reached] += 1

    odds = {}
    for team, stage_counts in counts.items():
        odds[team] = {stage: stage_counts.get(stage, 0) / n_sims for stage in STAGE_ORDER}
    return odds
