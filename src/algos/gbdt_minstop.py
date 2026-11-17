"""gbdt_weighted + a minimum stoppage length.

Two of the strongest public teams found that their Viterbi decoder sometimes inserts a dead ball of
a few frames inside live play, which splits one possession into two (or buys a cheap team change).
Barca-Hackathon: a 1 s minimum stoppage, worth 0.061 to them; PerosinhoFC: a stoppage decoder with
a 1.4 s minimum. Here: after decoding, any `D` run shorter than `min_dead_s` that sits between live
frames is given back to live play -- to the team before it (the change, if any, moves to its end).
Runs touching the start or end of a half are left alone.
"""
import numpy as np

from src.algos import _features as F
from src.algos import gbdt_weighted as GW

PARAMS = dict(GW.PARAMS, min_dead_s=1.0)
fit = GW.fit
STATS = {"blips": 0, "frames": 0}


def predict(match, model=None, min_dead_s=1.0, **params):
    labels = np.asarray(GW.predict(match, model=model, **params)).copy()
    min_len = int(round(min_dead_s * F.FPS))
    if min_len <= 0:
        return labels
    for _, first, last in match.halves:
        seg = labels[first:last + 1]
        for s, e in F.runs(seg == "D"):
            if s == 0 or e == len(seg) or e - s >= min_len:
                continue
            if any(first + s < b <= first + e for b, _ in getattr(match, "dead_spans", [])):
                continue
            seg[s:e] = seg[s - 1]
            STATS["blips"] += 1
            STATS["frames"] += e - s
    return labels
