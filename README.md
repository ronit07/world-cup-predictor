# World Cup Predictor

A side project where I tried to predict who wins the 2026 World Cup. It
gives every team an Elo rating, uses a Dixon-Coles model to get the
probability of each scoreline, and then simulates the whole 48-team
tournament 10,000 times to see how often each team gets to each round.

Part 2 (further down) retrains everything on ~50,000 real international
results, adds an XGBoost model, and serves predictions as an API on AWS
Lambda.

Here's what it spits out:

```
Simulated 10,000 tournaments in 4.2s (seed 42).
Each column is the chance of reaching at least that stage.

  #  Team           Grp     Elo   Win it    Final    Semis       QF      R16      R32
-------------------------------------------------------------------------------------
  1  Argentina      A      2010    12.1%    19.4%    30.4%    45.1%    66.6%    92.7%
  2  France         B      1995    10.0%    17.3%    28.1%    43.7%    65.4%    92.7%
  3  Brazil         C      1985     9.2%    15.6%    26.2%    42.0%    63.6%    92.7%
  4  Spain          E      1955     7.4%    13.6%    23.8%    39.5%    62.3%    92.5%
  5  England        D      1960     7.3%    13.5%    23.6%    39.9%    62.5%    92.2%
  ...
```

Note: the ratings and groups in `data/` are made-up demo values I put
together, not the real ratings or the official draw (see the data section
below if you want to plug in real ones).

## Running it

You just need Python 3.9+. The predictor only uses the standard library so
there's nothing to install.

```bash
git clone https://github.com/ronit07/world-cup-predictor.git
cd world-cup-predictor
python run_simulation.py
```

(use `python3` if `python` doesn't work on your machine)

Some other things you can do:

```bash
python run_simulation.py --all                     # show all 48 teams instead of top 20
python run_simulation.py --team USA Mexico Canada  # only show certain teams
python run_simulation.py --sims 50000 --seed 7     # more sims / different random seed

python src/backtest.py                 # check how accurate the model is
python src/backtest.py --cold-start    # same thing but every team starts at 1500
python src/generate_sample_data.py     # rebuild the sample match history

pip install -r requirements.txt        # only needed for the tests
pytest
```

Every script has a `--help` if you forget the options. If you keep the same
seed you get the same numbers every time.

## How it works

There are basically 4 parts:

**1. Elo ratings** (`src/elo.py`)
Every team has a rating. The difference between two teams' ratings (plus
home advantage if it's not a neutral venue) tells you who's expected to
win. After each game both ratings get updated, and bigger games and bigger
wins move them more. I used the same formula as the World Football Elo
Ratings site.

**2. Dixon-Coles model** (`src/dixon_coles.py`)
This turns the rating gap into expected goals for each team and then gives
a probability for every scoreline from 0-0 up to 8-8. A normal Poisson
model doesn't predict enough low scoring draws (0-0, 1-1), which is a
problem in football since those happen a lot, so Dixon and Coles (1997)
added a small correction for those scores.

**3. Monte Carlo simulation** (`src/simulate.py`)
For every match it picks a random scoreline based on those probabilities.
It plays all 12 groups, takes the top 2 from each group plus the 8 best
third place teams (that's the actual 2026 format), and then runs a 32-team
knockout bracket with penalties if it's a draw. Do that 10,000 times and
count how often each team makes it to each round.

**4. Backtest** (`src/backtest.py`)
To check if the model is any good, it goes through old matches in date
order and predicts each one *before* updating the ratings with the result,
so it can't cheat by already knowing the score.

## Part 2: real data + XGBoost

The first version only ran on made-up sample data, so the accuracy number
didn't really mean anything. So I switched to real results and added a
machine learning model on top.

**Data.** [martj42/international_results](https://github.com/martj42/international_results)
(CC0) has every men's international match since 1872, almost 50,000 games.
I copied `results.csv` into `data/raw/` and `src/real_data.py` converts it into
the same format the backtest already uses (friendly / qualifier /
continental / World Cup, plus neutral venue and host country).

**Features** (`src/features.py`). I go through every match in date order and
record the features *before* the result is known, then update Elo and form
afterwards, so nothing leaks from the future. For each match:

- both teams' Elo and the gap between them
- the Dixon-Coles win/draw/loss probabilities from part 1
- points per game and goal difference over the last 5 and last 10 games
- neutral venue, whether either team is the host, how important the game is

**Model** (`src/train_xgb.py`). A multiclass XGBoost classifier that outputs
win/draw/loss probabilities. Split by date:

| | dates | matches |
|---|---|---|
| train | 1990 - 2015 | 22,347 |
| validation (early stopping only) | 2016 - 2017 | 1,844 |
| test (never touched while training) | Jan 2018 - Oct 2026 | 8,630 |

Matches before 1990 are only used to warm up the ratings.

## How accurate is it?

On the 8,630 test matches:

| model | accuracy (no draws) | accuracy (3-way) | log-loss | Brier |
|---|---|---|---|---|
| always guess the base rates | 61.9% | 47.7% | 1.051 | 0.634 |
| Elo + Dixon-Coles (part 1) | 78.2% | 60.3% | 0.888 | 0.521 |
| XGBoost | 78.2% | 60.2% | **0.870** | **0.512** |

World Cup finals only (2018, 2022 and 2026, 232 matches):

| model | accuracy (no draws) | accuracy (3-way) | log-loss | Brier |
|---|---|---|---|---|
| Elo + Dixon-Coles | 76.7% | 59.5% | 0.959 | 0.567 |
| XGBoost | 73.9% | 57.3% | **0.947** | **0.559** |

Accuracy (no draws) = out of the games that weren't draws, how often the team
the model favoured actually won. Log-loss and Brier score measure how good the
probabilities are, lower is better.

Honestly the interesting part is that XGBoost doesn't pick more winners than
plain Elo, it's tied overall and a bit worse on World Cup games. What it does
do is give better probabilities (lower log-loss and Brier on both test sets),
mostly because the form features help it tell when a draw is likely. Elo is a
really strong baseline for international football. Draws are still the hard
part, the 3-way accuracy is stuck around 60%.

To reproduce:

```
pip install -r requirements.txt
python src/train_xgb.py
```

It saves the model to `models/xgb_model.json`, the numbers above to
`models/metrics.json`, and every team's latest Elo + form to
`models/team_features.json`.

(The old synthetic-data backtest from part 1 still works with
`python src/backtest.py`, it got 67.4% on the fake data.)

## Serving it as an API on AWS

`src/api.py` is a Lambda handler behind API Gateway:

```
GET /predict?home=Brazil&away=France&neutral=true&stage=world_cup_group
```

```json
{"home": "Brazil", "away": "France", "neutral": true, "stage": "world_cup_group",
 "probabilities": {"home_win": 0.41, "draw": 0.27, "away_win": 0.32}}
```

- the model file is stored in **S3** and loaded once per cold start, then
  kept in memory for warm requests
- each team's current Elo and form live in **DynamoDB** (one item per team)
- **CloudWatch** alarms on errors, p95 duration over 1s, and throttles
  (`deploy/template.yaml`)
- it runs as a container image because the normal xgboost package is too big
  for a zip Lambda, the image uses `xgboost-cpu`

Deploying (needs the AWS SAM CLI and Docker):

```
cd deploy
sam build
sam deploy --guided
python ../scripts/load_team_features.py --table <TeamTableName> --bucket <ModelBucketName>
```

Without AWS env vars the handler falls back to the files in `models/`, which
is how the tests run.

## Tests and CI

`pytest` runs everything, including `tests/test_api.py` which uses moto to
fake S3 and DynamoDB. A couple of tests I care about the most:

- `test_features_only_use_past_results` changes the score of a match and
  checks its features don't change (no leakage)
- `test_api_features_match_training_features` checks the API builds the
  exact same feature vector as training, otherwise the deployed model would
  silently get inputs in the wrong order

GitHub Actions runs the tests on every push (`.github/workflows/tests.yml`).

## Data (part 1 simulator)


Everything is in `data/` and you can edit it:

- `seed_ratings.csv` - starting Elo for each team. You could swap in real
  ratings from eloratings.net.
- `groups.json` - the 12 groups. Swap in the real draw if you want, just
  make sure every team is also in `seed_ratings.csv`.
- `sample_historical_matches.csv` - this is **fake** match history I
  generated with `src/generate_sample_data.py` so the backtest has
  something to run on. It's not real results.

If you want to backtest on real games, make a CSV with these columns:
`date,home_team,away_team,home_goals,away_goals,stage,neutral`
and run:

```bash
python src/backtest.py --data path/to/your_results.csv
```

`stage` can be `friendly`, `qualifier`, `continental`, `world_cup_group` or
`world_cup_knockout`, and `neutral` is `True` or `False`.

## Limitations / stuff I'd like to fix

- The knockout bracket gets shuffled randomly each time instead of using
  FIFA's actual bracket, so two of the best teams can run into each other
  in the round of 32.
- USA, Mexico and Canada don't get home advantage even though they're
  hosting. Every World Cup game is treated as neutral right now.
- Penalties are slightly weighted towards the better team, but in reality
  shootouts are closer to 50/50.
- Ratings don't change during a simulated tournament.
- The tournament simulator still uses the demo ratings and groups in
  `data/`. Next step is to feed it the real Elo from `team_features.json`.
- The raw data doesn't say if a World Cup game is group stage or knockout,
  so they're all weighted the same.

## Files

```
run_simulation.py           runs the tournament simulation and prints the odds
src/
  elo.py                    Elo ratings
  dixon_coles.py            scoreline model
  simulate.py               group stage + knockouts
  backtest.py               accuracy check (part 1)
  real_data.py              converts the real results dataset
  features.py               walk-forward features for every match
  train_xgb.py              trains + evaluates XGBoost vs Elo
  api.py                    Lambda handler
  generate_sample_data.py   makes the fake match history (part 1)
  data_loading.py           loads the files in data/
scripts/
  load_team_features.py     pushes team features to DynamoDB, model to S3
deploy/
  template.yaml             SAM template (Lambda, API Gateway, DynamoDB, S3, alarms)
  Dockerfile                Lambda container image
models/                     trained model, metrics, team features
data/
  raw/results.csv           real international results (CC0)
  seed_ratings.csv, groups.json, sample_historical_matches.csv
tests/                      pytest tests
```
