import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elo import EloRatingSystem
from simulate import simulate_group, simulate_knockout_bracket, run_monte_carlo


def _twelve_groups():
    """12 groups of 4 (48 teams), matching the real 2026 format so the
    32-team knockout bracket math works out (24 group winners/runners-up +
    8 best third-place teams = 32 = a power of two)."""
    groups = {}
    for g in range(12):
        letter = chr(ord("A") + g)
        groups[f"Group {letter}"] = [f"Team{g}{n}" for n in range(4)]
    return groups


def test_simulate_group_produces_four_standings_summing_points_correctly():
    elo = EloRatingSystem()
    rng = random.Random(1)
    standings = simulate_group(["A", "B", "C", "D"], elo, rng)
    assert len(standings) == 4
    total_points = sum(s.points for s in standings)
    # 6 matches per group; each contributes either 3 (decisive) or 2 (draw) points total
    assert 12 <= total_points <= 18


def test_knockout_bracket_rejects_non_power_of_two():
    elo = EloRatingSystem()
    rng = random.Random(1)
    try:
        simulate_knockout_bracket(["A", "B", "C"], elo, rng)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_knockout_bracket_produces_single_champion():
    elo = EloRatingSystem()
    rng = random.Random(2)
    teams = [f"T{i}" for i in range(32)]
    bracket = simulate_knockout_bracket(teams, elo, rng)
    assert len(bracket["champion"]) == 1
    assert len(bracket["final"]) == 2
    assert len(bracket["semifinal"]) == 4
    assert len(bracket["quarterfinal"]) == 8
    assert len(bracket["round_of_16"]) == 16


def test_run_monte_carlo_champion_probabilities_sum_close_to_one():
    groups = _twelve_groups()
    elo = EloRatingSystem()
    for team in [t for g in groups.values() for t in g]:
        elo.set(team, 1700)
    odds = run_monte_carlo(groups, elo, n_sims=100, seed=1)
    champion_prob_sum = sum(stage_odds["champion"] for stage_odds in odds.values())
    assert abs(champion_prob_sum - 1.0) < 1e-6


def test_stronger_team_wins_title_more_often():
    groups = _twelve_groups()
    elo = EloRatingSystem()
    for team in [t for g in groups.values() for t in g]:
        elo.set(team, 1700)
    elo.set("Team00", 2200)  # clearly dominant, in Group A
    odds = run_monte_carlo(groups, elo, n_sims=300, seed=3)
    assert odds["Team00"]["champion"] > odds["Team113"]["champion"]
