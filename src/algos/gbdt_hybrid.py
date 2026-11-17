"""gbdt_viterbi + the rule-based tracker's state as extra features.

The control_tracker (control radius + hysteresis + loose-ball rule) is a strong, physically
motivated possession estimate on its own. Here its per-frame output -- team in possession (+1 home
/ -1 away), dead-ball flag, seconds since its last team change and until its next one -- is
appended to the team-level features, so the classifier learns when to trust the rule and when the
context (team shape, ball gaps, speeds) says otherwise. Same mirroring, same Viterbi decode.
"""
import numpy as np

from src.algos import _features as F
from src.algos import _learned
from src.algos import control_tracker as CT
from src.algos import gbdt_viterbi as G

PARAMS = dict(G.PARAMS)
RULE = dict(CT.PARAMS)
TAG = "gbdt_hybrid"
_cache = {}
EXTRA = ["rule_team", "rule_dead", "rule_since", "rule_until", "rule_team_m125", "rule_dead_m125"]


def featurize(match):
    if match.game not in _cache:
        X, names = F.frame_features(match)
        ctrl, col = CT.control(match, RULE["radius_m"], RULE["margin_m"])
        dead = CT.dead_mask(match, col, RULE["dead_gap_s"], RULE["dead_speed"], RULE["out_m"],
                            RULE["min_dead_s"])
        team = CT.possession(ctrl, dead, int(round(RULE["hold_s"] * F.FPS)))
        rule_team = np.where(team == 0, 1.0, -1.0)
        i = np.arange(len(team))
        change = np.flatnonzero(np.diff(team) != 0) + 1
        since = np.full(len(team), np.nan)
        until = np.full(len(team), np.nan)
        if len(change):
            prev = np.searchsorted(change, i, side="right") - 1
            nxt = np.searchsorted(change, i, side="right")
            since = np.where(prev >= 0, (i - change[np.maximum(prev, 0)]) / F.FPS, np.nan)
            until = np.where(nxt < len(change),
                             (change[np.minimum(nxt, len(change) - 1)] - i) / F.FPS, np.nan)
        extra = np.stack([rule_team, dead.astype(float), since, until,
                          F.rolling_mean(rule_team, 125), F.rolling_mean(dead.astype(float), 125)], 1)
        _cache.clear()  # one match at a time is enough
        _cache[match.game] = (np.concatenate([X, extra.astype(np.float32)], 1), names + EXTRA)
    return _cache[match.game]


def mirror(X, names):
    Y = F.mirror(X, names)
    for n in ("rule_team", "rule_team_m125"):
        j = names.index(n)
        Y[:, j] = -X[:, j]
    return Y


def fit(matches, **params):
    return _learned.fit(TAG, featurize, mirror, matches, **params)


def predict(match, model=None, switch_team=8.0, switch_dead=6.0, **_):
    if model is None:
        raise ValueError("needs a fitted model (src.run and the bench fit it)")
    P = _learned.probabilities(featurize, mirror, match, model)
    return _learned.decode(match, P, switch_team, switch_dead)
