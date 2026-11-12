"""Upload models/team_features.json into the DynamoDB team table and the
model file into S3. Run this after train_xgb.py and after the stack is
deployed.

Usage:
    python scripts/load_team_features.py --table <TeamTable> --bucket <ModelBucket>
"""

import argparse
import json
import os
from decimal import Decimal

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--key", default="xgb_model.json")
    args = parser.parse_args()

    with open(os.path.join(MODELS, "team_features.json"), encoding="utf-8") as f:
        teams = json.load(f)

    table = boto3.resource("dynamodb").Table(args.table)
    with table.batch_writer() as batch:
        for team, feats in teams.items():
            item = {"team": team}
            item.update({k: Decimal(str(v)) for k, v in feats.items()})
            batch.put_item(Item=item)
    print(f"wrote {len(teams)} teams to {args.table}")

    boto3.client("s3").upload_file(os.path.join(MODELS, "xgb_model.json"), args.bucket, args.key)
    print(f"uploaded model to s3://{args.bucket}/{args.key}")


if __name__ == "__main__":
    main()
