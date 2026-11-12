"""Train an XGBoost win/draw/loss model on real international results and
compare it against the original Elo + Dixon-Coles model on the same
held-out matches.

Split (by date, so there's no leakage from the future):
  train: 1990-01-01 to 2015-12-31
  valid: 2016-01-01 to 2017-12-31  (only used for early stopping)
  test:  2018-01-01 onwards        (never touched during training)

Everything before 1990 is only used to warm up the Elo ratings and form.

Usage:
    python src/real_data.py      # once, builds data/real_matches.csv
    python src/train_xgb.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import xgboost as xgb

from features import FEATURE_NAMES, build_dataset, load_real_matches

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "real_matches.csv")
MODEL_DIR = os.path.join(HERE, "..", "models")

TRAIN_START, VALID_START, TEST_START = "1990-01-01", "2016-01-01", "2018-01-01"


def split(rows):
    train = [r for r in rows if TRAIN_START <= r["date"] < VALID_START]
    valid = [r for r in rows if VALID_START <= r["date"] < TEST_START]
    test = [r for r in rows if r["date"] >= TEST_START]
    return train, valid, test


def to_xy(rows):
    X = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=int)
    return X, y


def evaluate(probs: np.ndarray, y: np.ndarray) -> dict:
    """Same accuracy definition as backtest.py: of the matches that weren't
    draws, how often the favoured team (home vs away) actually won."""
    eps = 1e-15
    n = len(y)
    log_loss = -sum(math.log(max(probs[i, y[i]], eps)) for i in range(n)) / n
    onehot = np.eye(3)[y]
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    decisive = y != 1
    fav_home = probs[:, 0] >= probs[:, 2]
    home_won = y == 0
    acc_decisive = float(np.mean((fav_home == home_won)[decisive]))
    acc_3way = float(np.mean(np.argmax(probs, axis=1) == y))
    return {
        "matches": int(n),
        "decisive_matches": int(decisive.sum()),
        "accuracy_decisive": round(acc_decisive, 4),
        "accuracy_3way": round(acc_3way, 4),
        "log_loss": round(log_loss, 4),
        "brier": round(brier, 4),
    }


def main() -> None:
    if not os.path.exists(DATA_PATH):
        from real_data import convert

        convert()
    matches = load_real_matches(DATA_PATH)
    rows, elo, form = build_dataset(matches)
    train, valid, test = split(rows)
    X_tr, y_tr = to_xy(train)
    X_va, y_va = to_xy(valid)
    X_te, y_te = to_xy(test)
    print(f"train {len(train)}  valid {len(valid)}  test {len(test)}  (test = {TEST_START} onwards)")

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        early_stopping_rounds=100,
        random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"best iteration: {model.best_iteration}")

    xgb_probs = model.predict_proba(X_te)
    dc_probs = np.array([r["dc_probs"] for r in test])
    # Dixon-Coles probabilities don't always sum to exactly 1 because the
    # scoreline grid is capped, so normalise before scoring
    dc_probs = dc_probs / dc_probs.sum(axis=1, keepdims=True)
    base = np.bincount(y_tr, minlength=3) / len(y_tr)
    base_probs = np.tile(base, (len(y_te), 1))

    # World Cup finals only (2018 + 2022), the hardest and most relevant games
    wc = np.array([r["features"][-1] >= 4 for r in test])

    results = {
        "test_period": f"{test[0]['date']} to {test[-1]['date']}",
        "xgboost": evaluate(xgb_probs, y_te),
        "elo_dixon_coles": evaluate(dc_probs, y_te),
        "base_rates": evaluate(base_probs, y_te),
        "world_cup_only": {
            "xgboost": evaluate(xgb_probs[wc], y_te[wc]),
            "elo_dixon_coles": evaluate(dc_probs[wc], y_te[wc]),
        },
        "best_iteration": int(model.best_iteration),
        "features": FEATURE_NAMES,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # latest Elo + form for every team, this is what goes into DynamoDB
    teams = {}
    for team in elo.ratings:
        ppg5, gd5 = form.stats(team, 5)
        ppg10, gd10 = form.stats(team, 10)
        teams[team] = {
            "elo": round(elo.get(team), 2),
            "ppg_5": round(ppg5, 4),
            "gd_5": round(gd5, 4),
            "ppg_10": round(ppg10, 4),
            "gd_10": round(gd10, 4),
            "games_played": min(form.games_played[team], 50),
        }
    with open(os.path.join(MODEL_DIR, "team_features.json"), "w") as f:
        json.dump(teams, f, indent=1, sort_keys=True)

    print()
    print(f"{'model':<18}{'acc (no draws)':>16}{'acc (3-way)':>13}{'log-loss':>10}{'brier':>8}")
    for name in ("base_rates", "elo_dixon_coles", "xgboost"):
        r = results[name]
        print(f"{name:<18}{r['accuracy_decisive']:>16.1%}{r['accuracy_3way']:>13.1%}{r['log_loss']:>10.4f}{r['brier']:>8.4f}")
    print(f"\nWorld Cup finals only ({results['world_cup_only']['xgboost']['matches']} matches):")
    for name in ("elo_dixon_coles", "xgboost"):
        r = results["world_cup_only"][name]
        print(f"{name:<18}{r['accuracy_decisive']:>16.1%}{r['accuracy_3way']:>13.1%}{r['log_loss']:>10.4f}{r['brier']:>8.4f}")


if __name__ == "__main__":
    main()
