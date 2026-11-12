"""Tests for the Lambda handler. moto fakes S3 and DynamoDB so nothing here
touches a real AWS account."""

import json
import os
import sys
from decimal import Decimal

import boto3
import numpy as np
import pytest
import xgboost as xgb
from moto import mock_aws

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import api  # noqa: E402
from features import FEATURE_NAMES, match_features, EloRatingSystem, FormTracker  # noqa: E402

TEAMS = {
    "Brazil": {"elo": 2050.0, "ppg_5": 2.4, "gd_5": 1.6, "ppg_10": 2.2, "gd_10": 1.3, "games_played": 50},
    "Andorra": {"elo": 1050.0, "ppg_5": 0.2, "gd_5": -2.8, "ppg_10": 0.3, "gd_10": -2.5, "games_played": 50},
}


@pytest.fixture
def tiny_model(tmp_path):
    """Small model trained on fake data where a bigger Elo gap means the home
    team wins, so the API output has a direction I can check."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, len(FEATURE_NAMES)))
    X[:, 2] = rng.normal(0, 300, 600)  # elo_diff
    y = np.where(X[:, 2] > 150, 0, np.where(X[:, 2] < -150, 2, 1))
    params = {"objective": "multi:softprob", "num_class": 3, "max_depth": 2}
    booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=30)
    path = tmp_path / "xgb_model.json"
    booster.save_model(path)
    return path


@pytest.fixture
def aws(tiny_model, monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        s3 = boto3.client("s3")
        s3.create_bucket(Bucket="models", CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
        s3.upload_file(str(tiny_model), "models", "xgb_model.json")
        ddb = boto3.resource("dynamodb")
        table = ddb.create_table(
            TableName="teams",
            KeySchema=[{"AttributeName": "team", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "team", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        for team, feats in TEAMS.items():
            table.put_item(Item={"team": team, **{k: Decimal(str(v)) for k, v in feats.items()}})
        monkeypatch.setenv("MODEL_BUCKET", "models")
        monkeypatch.setenv("TEAM_TABLE", "teams")
        monkeypatch.setattr(api, "_model", None)
        yield


def call(**params):
    resp = api.handler({"queryStringParameters": params}, None)
    return resp["statusCode"], json.loads(resp["body"])


def test_prediction_from_s3_and_dynamodb(aws):
    status, body = call(home="Brazil", away="Andorra")
    assert status == 200
    p = body["probabilities"]
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1) < 1e-3
    assert p["home_win"] > p["away_win"]


def test_swapping_teams_flips_the_favourite(aws):
    _, body = call(home="Andorra", away="Brazil")
    assert body["probabilities"]["away_win"] > body["probabilities"]["home_win"]


def test_model_is_cached_between_calls(aws):
    call(home="Brazil", away="Andorra")
    first = api._model
    call(home="Andorra", away="Brazil")
    assert api._model is first


@pytest.mark.parametrize(
    "params, status",
    [
        ({"home": "Brazil"}, 400),
        ({"home": "Brazil", "away": "Brazil"}, 400),
        ({"home": "Brazil", "away": "Andorra", "stage": "final_boss"}, 400),
        ({"home": "Brazil", "away": "Narnia"}, 404),
    ],
)
def test_bad_requests(aws, params, status):
    code, body = call(**params)
    assert code == status
    assert "error" in body


def test_api_features_match_training_features():
    """api.build_features has to produce the same vector as features.py, or
    the deployed model would get inputs in the wrong order."""
    elo, form = EloRatingSystem(ratings={"X": 1800.0, "Y": 1600.0}), FormTracker()
    for gf, ga in [(2, 0), (1, 1), (0, 1), (3, 2), (1, 0), (2, 2)]:
        form.add("X", gf, ga)
        form.add("Y", ga, gf)
    train_vec = match_features(elo, form, "X", "Y", True, "Qatar", "world_cup_group")

    def team(t):
        ppg5, gd5 = form.stats(t, 5)
        ppg10, gd10 = form.stats(t, 10)
        return {"elo": elo.get(t), "ppg_5": ppg5, "gd_5": gd5, "ppg_10": ppg10,
                "gd_10": gd10, "games_played": form.games_played[t]}

    api_vec = api.build_features(team("X"), team("Y"), True, False, False, "world_cup_group")
    assert api_vec == pytest.approx(train_vec)


def test_local_fallback_without_aws(monkeypatch):
    """With no env vars it uses the committed model and team file."""
    monkeypatch.delenv("MODEL_BUCKET", raising=False)
    monkeypatch.delenv("TEAM_TABLE", raising=False)
    monkeypatch.setattr(api, "_model", None)
    status, body = call(home="Argentina", away="France")
    assert status == 200
