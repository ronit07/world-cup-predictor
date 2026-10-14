# World Cup 2026 Predictor

A side project where I tried to predict who wins the 2026 World Cup. It
gives every team an Elo rating, uses a Dixon-Coles model to get the
probability of each scoreline, and then simulates the whole 48-team
tournament 10,000 times to see how often each team gets to each round.

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

## How accurate is it?

Results on the 600 sample matches:

| | Accuracy | Brier score |
|---|---|---|
| Just guessing 1/3 for each outcome | - | 0.667 |
| Every team starts at 1500 (`--cold-start`) | 58.8% | 0.631 |
| Starting from the seed ratings (default) | 67.4% | 0.604 |

Accuracy = out of the games that weren't draws, how often the team the
model favoured actually won (50% would be a coin flip).

Brier score = how close the predicted win/draw/loss probabilities were to
what actually happened. Lower is better. It's never going to get near 0
because football is pretty random, and even the "perfect" model (the exact
one that generated the sample data) only gets 69.7% and 0.588.

## Data

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

## Files

```
run_simulation.py           runs the simulation and prints the odds
src/
  elo.py                    Elo ratings
  dixon_coles.py            scoreline model
  simulate.py               group stage + knockouts
  backtest.py               accuracy check
  generate_sample_data.py   makes the fake match history
  data_loading.py           loads the files in data/
data/
  seed_ratings.csv
  groups.json
  sample_historical_matches.csv
tests/                      pytest tests
```
