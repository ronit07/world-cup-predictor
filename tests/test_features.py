import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import FEATURE_NAMES, FormTracker, build_dataset, outcome_label
from real_data import stage_for


def _match(date, home, away, hg, ag, stage="friendly", neutral=False, country=""):
    return {
        "date": date, "home": home, "away": away, "home_goals": hg, "away_goals": ag,
        "stage": stage, "neutral": neutral, "country": country or home,
    }


def test_features_only_use_past_results():
    """A match's features can't depend on its own score. Changing the result
    of the last match must not change the features recorded for it."""
    base = [_match("2020-01-01", "A", "B", 1, 0), _match("2020-02-01", "A", "B", 0, 0)]
    flipped = [base[0], _match("2020-02-01", "A", "B", 5, 0)]
    rows_a, _, _ = build_dataset(base)
    rows_b, _, _ = build_dataset(flipped)
    assert rows_a[-1]["features"] == rows_b[-1]["features"]
    assert rows_a[-1]["label"] != rows_b[-1]["label"]


def test_first_match_uses_default_ratings_and_form():
    rows, _, _ = build_dataset([_match("2020-01-01", "A", "B", 2, 1)])
    feats = dict(zip(FEATURE_NAMES, rows[0]["features"]))
    assert feats["elo_home"] == feats["elo_away"] == 1500
    assert feats["home_games_played"] == 0
    assert feats["home_is_host"] == 1.0 and feats["away_is_host"] == 0.0


def test_winner_gains_elo_and_form():
    rows, elo, form = build_dataset([_match("2020-01-01", "A", "B", 3, 0)])
    assert elo.get("A") > 1500 > elo.get("B")
    assert form.stats("A", 5) == (3.0, 3.0)
    assert form.stats("B", 5) == (0.0, -3.0)


def test_form_window_keeps_last_ten():
    form = FormTracker()
    for _ in range(12):
        form.add("A", 1, 0)
    form.add("A", 0, 0)
    assert len(form.history["A"]) == 10
    ppg5, _ = form.stats("A", 5)
    assert abs(ppg5 - (4 * 3 + 1) / 5) < 1e-9


def test_outcome_labels():
    assert outcome_label(2, 1) == 0
    assert outcome_label(1, 1) == 1
    assert outcome_label(0, 3) == 2


def test_stage_mapping():
    assert stage_for("Friendly") == "friendly"
    assert stage_for("FIFA World Cup") == "world_cup_group"
    assert stage_for("FIFA World Cup qualification") == "qualifier"
    assert stage_for("UEFA Euro") == "continental"
    assert stage_for("Island Games") == "other"
