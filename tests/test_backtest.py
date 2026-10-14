import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtest import load_matches, uniform_brier, walk_forward_backtest
from data_loading import SAMPLE_MATCHES_PATH, load_groups, load_seed_ratings
from dixon_coles import match_probabilities
from elo import EloRatingSystem


def test_seeded_backtest_beats_cold_start_and_guessing():
    """The sample data is generated from the seed ratings, so starting from
    them should score better than learning every rating from 1500."""
    matches = load_matches(SAMPLE_MATCHES_PATH)
    seeded = walk_forward_backtest(matches, EloRatingSystem(ratings=load_seed_ratings()))
    cold = walk_forward_backtest(matches, EloRatingSystem())
    assert seeded["brier_score"] < cold["brier_score"] < uniform_brier(matches)
    assert seeded["accuracy_decisive"] > cold["accuracy_decisive"] > 0.5


def test_uniform_brier_is_two_thirds():
    matches = load_matches(SAMPLE_MATCHES_PATH)
    assert abs(uniform_brier(matches) - 2 / 3) < 1e-9


def test_every_group_team_has_a_seed_rating():
    ratings = load_seed_ratings()
    teams = [t for group in load_groups().values() for t in group]
    assert len(teams) == 48 == len(set(teams))
    assert all(t in ratings for t in teams)


def test_cached_match_probabilities_are_consistent_and_read_only():
    first = match_probabilities(1900, 1750, home_advantage=100.0)
    second = match_probabilities(1900.0, 1750.0, home_advantage=100)
    assert first["scoreline_matrix"] is second["scoreline_matrix"]  # served from the cache
    assert first["home_win"] == second["home_win"]
    try:
        first["scoreline_matrix"][0][0] = 1.0
        assert False, "cached matrix should be immutable"
    except TypeError:
        pass
