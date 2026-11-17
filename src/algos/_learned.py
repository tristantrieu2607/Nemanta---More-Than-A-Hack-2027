"""Shared machinery for learned algorithms (not an algorithm): fit a per-frame H/A/D classifier on
external matches, predict probabilities (averaged with the home/away-mirrored view), decode.

An algorithm supplies `featurize(match) -> (X, names)` and `mirror(X, names) -> X'`; see
gbdt_viterbi.py. Models, training rows and probabilities are cached in memory per training set /
match, so sweeping decode parameters (switch costs) does not refit anything.
"""
import hashlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.algos import _features as F

SEED = 2027
_rows, _models, _probs = {}, {}, {}


def _training_rows(tag, featurize, match, stride, reading="main"):
    key = (tag, match.game, stride, reading)
    if key not in _rows:
        X, names = featurize(match)
        if reading == "mix":  # half the frames labelled under each reading of "under control"
            y = match.gt("main")
            y_short = match.gt("short")
            alt = np.arange(len(y)) % (2 * stride) >= stride
            y = np.where(alt, y_short, y)
        else:
            y = match.gt(reading)
        rng = np.random.default_rng(int(hashlib.md5(match.game.encode()).hexdigest()[:8], 16))
        idx = np.flatnonzero(match.graded & (y < 3))
        idx = idx[(idx + int(rng.integers(stride))) % stride == 0]
        _rows[key] = (X[idx], y[idx].astype(np.int8), names, _row_kind(y)[idx])
    return _rows[key]


def _row_kind(y, short_s=4.0, edge_frames=12):
    """Per frame: bit 1 = inside a possession shorter than `short_s`, bit 2 = within
    `edge_frames` of a label change. Used for the optional training weights."""
    change = np.flatnonzero(np.diff(y) != 0) + 1
    bounds = np.concatenate(([0], change, [len(y)]))
    kind = np.zeros(len(y), np.int8)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if y[a] < 2 and (b - a) < short_s * F.FPS:
            kind[a:b] |= 1
    for c in change:
        kind[max(0, c - edge_frames):c + edge_frames] |= 2
    return kind


def fit(tag, featurize, mirror, matches, stride=8, iters=300, lr=0.08, leaves=31, min_leaf=400,
        use_mirror=True, train_reading="main", short_weight=1.0, edge_weight=1.0, **_):
    fit_params = (stride, iters, lr, leaves, min_leaf, use_mirror, train_reading, short_weight,
                  edge_weight)
    names_ = getattr(matches, "names", None)
    key = hashlib.md5(repr((tag, names_, fit_params)).encode()).hexdigest()
    if names_ is not None and key in _models:
        return _models[key]
    Xs, ys, ws = [], [], []
    for match in matches:
        X, y, names, kind = _training_rows(tag, featurize, match, stride, train_reading)
        w = np.where(kind & 1, short_weight, 1.0) * np.where(kind & 2, edge_weight, 1.0)
        Xs.append(X)
        ys.append(y)
        ws.append(w)
        if use_mirror:
            Xs.append(mirror(X, names))
            ys.append(np.where(y == 0, 1, np.where(y == 1, 0, y)).astype(np.int8))
            ws.append(w)
    clf = HistGradientBoostingClassifier(max_iter=iters, learning_rate=lr, max_leaf_nodes=leaves,
                                         min_samples_leaf=min_leaf, l2_regularization=1.0,
                                         early_stopping=False, random_state=SEED)
    weights = np.concatenate(ws)
    clf.fit(np.concatenate(Xs), np.concatenate(ys),
            sample_weight=None if np.all(weights == 1.0) else weights)
    model = {"clf": clf, "names": names, "mirror": use_mirror, "key": key, "tag": tag}
    _models[key] = model
    return model


def probabilities(featurize, mirror, match, model):
    key = (model["key"], match.game)
    if key not in _probs:
        X, names = featurize(match)
        clf = model["clf"]
        cols = list(clf.classes_)
        P = np.zeros((len(X), 3))
        P[:, cols] = clf.predict_proba(X)
        if model["mirror"]:
            Pm = np.zeros((len(X), 3))
            Pm[:, cols] = clf.predict_proba(mirror(X, names))
            P = (P + Pm[:, [1, 0, 2]]) / 2
        _probs[key] = P.astype(np.float32)
    return _probs[key]


def decode(match, P, switch_team, switch_dead):
    return F.decode_halves(match, np.log(P + 1e-6), switch_team, switch_dead)
