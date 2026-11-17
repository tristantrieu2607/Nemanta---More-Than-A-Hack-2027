"""Learned per-frame P(H/A/D) from team-level features, decoded by Viterbi with switch costs.

The pattern every credible public entry converged on (Team Sigue, Agamemnon, Barca-Hackathon,
False9): no player identities, a classifier trained only on external matches degraded to look
like the ATD (src/external), then a 3-state Viterbi that charges `switch_team` (log units) for every
H<->A change and `switch_dead` for every alive<->dead change -- spellF1 punishes flicker hard, the
costs are what stops it.

Features: src/algos/_features.py (ball gaps, nearest players per team, team shape and speed,
attack direction from team centroids, rolling context up to 25 s, last/next confident controller).
Home/away symmetry: training rows are duplicated with teams swapped (`mirror`), and prediction
averages the model with its mirrored view, so the model cannot learn "home has the ball more".

Training (src/algos/_learned.py): sklearn HistGradientBoosting, fixed seed, every `stride`-th graded
frame, labels from the `main` reading. On BAR-ATH the model is fitted on all external matches.
"""
from src.algos import _features as F
from src.algos import _learned

PARAMS = {"switch_team": 8.0, "switch_dead": 6.0, "stride": 8, "iters": 300, "lr": 0.08,
          "leaves": 31, "min_leaf": 400, "use_mirror": True}
TAG = "gbdt_viterbi"


def fit(matches, **params):
    return _learned.fit(TAG, F.frame_features, F.mirror, matches, **params)


def predict(match, model=None, switch_team=8.0, switch_dead=6.0, **_):
    if model is None:
        raise ValueError("needs a fitted model (src.run and the bench fit it)")
    P = _learned.probabilities(F.frame_features, F.mirror, match, model)
    return _learned.decode(match, P, switch_team, switch_dead)
