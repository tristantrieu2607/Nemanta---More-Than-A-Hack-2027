"""gbdt_viterbi, then every H<->A change is snapped to the first touch of the new team.

The Viterbi path places a change where the probabilities cross, which tends to sit a little after
the real moment of control. spellF1 matches possessions by temporal IoU (1 s late ~ 0.65), so each
change between team X and team Y is moved to the first frame within `snap_back_s` before (or
`snap_fwd_s` after) it where a Y player has the ball under control (tracked ball, nearest Y
player within `radius_m` and closer than any X player). No control seen -> the change stays put.
Same model and caches as gbdt_viterbi.
"""
import numpy as np

from src.algos import _features as F
from src.algos import _learned
from src.algos import gbdt_viterbi as G

PARAMS = dict(G.PARAMS, snap_back_s=2.0, snap_fwd_s=0.5, radius_m=1.5)
TAG = G.TAG  # same model as gbdt_viterbi


def fit(matches, **params):
    return G.fit(matches, **params)


def snap(labels, match, snap_back_s, snap_fwd_s, radius_m):
    X, names = F.frame_features(match)
    h, a = X[:, names.index("d1_h")], X[:, names.index("d1_a")]
    seen = match.ball_seen
    ctrl = np.full(len(labels), -1, np.int8)
    ctrl[seen & (h < radius_m) & (h < a)] = 0
    ctrl[seen & (a < radius_m) & (a < h)] = 1
    code = np.select([labels == "H", labels == "A"], [0, 1], -1)
    out = labels.copy()
    back, fwd = int(snap_back_s * F.FPS), int(snap_fwd_s * F.FPS)
    change = np.flatnonzero((code[1:] >= 0) & (code[:-1] >= 0) & (code[1:] != code[:-1])) + 1
    for t in change:
        new = code[t]
        lo = max(0, t - back)
        # do not cross the previous change or a dead ball
        prev_bad = np.flatnonzero(code[lo:t] != code[t - 1])
        if len(prev_bad):
            lo = lo + prev_bad[-1] + 1
        hit = np.flatnonzero(ctrl[lo:t + fwd + 1] == new)
        if not len(hit):
            continue
        f = lo + hit[0]
        letter = "H" if new == 0 else "A"
        if f < t:
            out[f:t] = letter
        elif f > t:
            nxt_bad = np.flatnonzero(code[t:f] != new)
            if not len(nxt_bad):
                out[t:f] = "H" if code[t - 1] == 0 else "A"
    return out


def predict(match, model=None, switch_team=8.0, switch_dead=6.0, snap_back_s=2.0, snap_fwd_s=0.5,
            radius_m=1.5, **_):
    if model is None:
        raise ValueError("needs a fitted model (src.run and the bench fit it)")
    P = _learned.probabilities(F.frame_features, F.mirror, match, model)
    labels = _learned.decode(match, P, switch_team, switch_dead)
    return snap(labels, match, snap_back_s, snap_fwd_s, radius_m)
