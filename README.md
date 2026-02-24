# Steph Curry Shot Probabilities (2022–2023) — Udacity / Woolf Nanodegree

This project analyzes Steph Curry’s shot data from the 2022–2023 NBA season and computes:

- Overall shooting probabilities (make/miss, 2pt/3pt proportions)
- Binomial probabilities for future shot sequences
- Conditional probabilities (future + retrospective) using Bayes' Theorem

## Dataset

The repository includes a Steph Curry shot dataset CSV in `data/steph_curry_shots_2023.csv`.

Expected columns (minimum used by this analysis):
- `result` (TRUE/FALSE)
- `shot_type` (2 or 3)
- `lead` (TRUE/FALSE)

## How to run

### Option A — Notebook
Open and run:
- `notebooks/01_probability_analysis.ipynb`

### Option B — Python module
```bash
pip install -r requirements.txt
python -m src.curry_probabilities --csv data/steph_curry_shots_2023.csv
```

## Notes on assumptions (Binomial)

We assume:
- Fixed number of trials
- Independent shots
- Same probability each shot
- Two outcomes (make/miss)

In reality, independence/stationarity can be violated due to defense pressure, fatigue, and game situation.

## Outputs
- Final slide deck: `outputs/udacity_slide_deck.pdf`
