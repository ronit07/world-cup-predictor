import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from elo import EloRatingSystem, expected_score, goal_diff_multiplier


def test_expected_score_symmetry():
    p_a = expected_score(1600, 1400)
    p_b = expected_score(1400, 1600)
    assert abs((p_a + p_b) - 1.0) < 1e-9
    assert p_a > p_b


def test_goal_diff_multiplier_monotonic():
    assert goal_diff_multiplier(0) == 1.0
    assert goal_diff_multiplier(1) == 1.0
    assert goal_diff_multiplier(2) == 1.5
    assert goal_diff_multiplier(4) > goal_diff_multiplier(3) > goal_diff_multiplier(2)


def test_winner_gains_rating_against_higher_rated_opponent():
    elo = EloRatingSystem()
    elo.set("Underdog", 1500)
    elo.set("Favourite", 1900)
    new_underdog, new_favourite = elo.update_match("Underdog", "Favourite", 2, 1, stage="world_cup_group")
    assert new_underdog > 1500
    assert new_favourite < 1900


def test_draw_between_equal_teams_is_neutral():
    elo = EloRatingSystem()
    elo.set("A", 1700)
    elo.set("B", 1700)
    new_a, new_b = elo.update_match("A", "B", 1, 1, neutral_site=True)
    assert abs(new_a - 1700) < 1e-6
    assert abs(new_b - 1700) < 1e-6
