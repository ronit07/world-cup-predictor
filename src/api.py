"""AWS Lambda handler that serves match predictions through API Gateway.

Request:
    GET /predict?home=Brazil&away=France&neutral=true&stage=world_cup_group

Response:
    {"home": "Brazil", "away": "France",
     "probabilities": {"home_win": 0.41, "draw": 0.27, "away_win": 0.32}}

How it works:
  - the trained XGBoost model lives in S3 (MODEL_BUCKET / MODEL_KEY). It gets
    downloaded once per cold start and kept in memory after that.
  - each team's current Elo and form are in a DynamoDB table (TEAM_TABLE),
    one item per team. scripts/load_team_features.py fills it after training.

Locally (no env vars set) it falls back to models/xgb_model.json and
models/team_features.json so I can test without AWS.
"""

from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal

import numpy as np

from dixon_coles import match_probabilities
from elo import HOME_ADVANTAGE
from features import STAGES

logger = logging.getLogger()
logger.setLevel(logging.INFO)

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL = os.path.join(HERE, "..", "models", "xgb_model.json")
LOCAL_TEAMS = os.path.join(HERE, "..", "models", "team_features.json")

_model = None  # cached between warm invocations
_local_teams = None


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps(body),
    }


def get_model():
    global _model
    if _model is not None:
        return _model
    import xgboost as xgb

    bucket = os.environ.get("MODEL_BUCKET")
    path = LOCAL_MODEL
    if bucket:
        import boto3

        path = "/tmp/xgb_model.json"
        boto3.client("s3").download_file(bucket, os.environ.get("MODEL_KEY", "xgb_model.json"), path)
    model = xgb.XGBClassifier()
    model.load_model(path)
    _model = model
    return _model


def get_team(team: str) -> dict | None:
    table_name = os.environ.get("TEAM_TABLE")
    if table_name:
        import boto3

        item = boto3.resource("dynamodb").Table(table_name).get_item(Key={"team": team}).get("Item")
        if item is None:
            return None
        # DynamoDB gives numbers back as Decimal
        return {k: float(v) if isinstance(v, Decimal) else v for k, v in item.items()}

    global _local_teams
    if _local_teams is None:
        with open(LOCAL_TEAMS, encoding="utf-8") as f:
            _local_teams = json.load(f)
    return _local_teams.get(team)


def build_features(home: dict, away: dict, neutral: bool, home_is_host: bool, away_is_host: bool, stage: str):
    """Same order as features.FEATURE_NAMES. If this ever drifts from
    features.py the model silently gets garbage, so test_api.py checks it."""
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    dc = match_probabilities(home["elo"], away["elo"], home_advantage=home_adv)
    return [
        home["elo"],
        away["elo"],
        home["elo"] - away["elo"],
        dc["home_win"],
        dc["draw"],
        dc["away_win"],
        home["ppg_5"],
        away["ppg_5"],
        home["gd_5"],
        away["gd_5"],
        home["ppg_10"],
        away["ppg_10"],
        home["gd_10"],
        away["gd_10"],
        home["games_played"],
        away["games_played"],
        float(neutral),
        float(home_is_host),
        float(away_is_host),
        float(STAGES.index(stage)),
    ]


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes")


def handler(event, context):
    start = time.perf_counter()
    params = (event or {}).get("queryStringParameters") or {}
    home_name = (params.get("home") or "").strip()
    away_name = (params.get("away") or "").strip()
    stage = (params.get("stage") or "world_cup_group").strip()

    if not home_name or not away_name:
        return _response(400, {"error": "home and away are required"})
    if home_name == away_name:
        return _response(400, {"error": "home and away must be different teams"})
    if stage not in STAGES:
        return _response(400, {"error": f"stage must be one of {STAGES}"})

    home, away = get_team(home_name), get_team(away_name)
    missing = [n for n, t in ((home_name, home), (away_name, away)) if t is None]
    if missing:
        return _response(404, {"error": f"unknown team(s): {', '.join(missing)}"})

    neutral = _as_bool(params.get("neutral"), default=True)
    feats = build_features(
        home,
        away,
        neutral,
        _as_bool(params.get("home_is_host"), False),
        _as_bool(params.get("away_is_host"), False),
        stage,
    )
    probs = get_model().predict_proba(np.array([feats], dtype=float))[0]

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(json.dumps({"home": home_name, "away": away_name, "latency_ms": round(elapsed_ms, 2)}))
    return _response(
        200,
        {
            "home": home_name,
            "away": away_name,
            "neutral": neutral,
            "stage": stage,
            "probabilities": {
                "home_win": round(float(probs[0]), 4),
                "draw": round(float(probs[1]), 4),
                "away_win": round(float(probs[2]), 4),
            },
        },
    )
