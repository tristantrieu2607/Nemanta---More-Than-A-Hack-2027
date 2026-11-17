"""gbdt_weighted + Tristan's match-watching observations (src/algos/_tristan.py).

Convergence on a loose ball, how long a team keeps close control, forward runs after a win, and
marking at restarts -- all from positions. Same training weights, decoder and kickoff rule as
gbdt_weighted.
"""
import numpy as np

from src.algos import _features as F
from src.algos import _learned
from src.algos import _tristan as T
from src.algos import gbdt_plus as GP
from src.algos import gbdt_weighted as GW

PARAMS = dict(GW.PARAMS)
_cache = {}


def featurize(match, groups=PARAMS["groups"]):
    key = (groups, match.game)
    if key not in _cache:
        X, names = GP.featurizer(groups)(match)
        Y, yn = T.tristan_features(match)
        _cache.clear()
        _cache[key] = (np.concatenate([X, Y], 1), list(names) + yn)
    return _cache[key]


def mirror(X, names):
    return GP.mirror(X, names)


def fit(matches, groups=PARAMS["groups"], **params):
    return _learned.fit("gbdt_tristan:" + groups, lambda m: featurize(m, groups), mirror, matches, **params)


def predict(match, model=None, groups=PARAMS["groups"], switch_team=16.0, switch_dead=12.0,
            dead_bias=0.0, **_):
    if model is None:
        raise ValueError("needs a fitted model (src.run and the bench fit it)")
    P = _learned.probabilities(lambda m: featurize(m, groups), mirror, match, model)
    logp = np.log(P + 1e-6)
    logp[:, 2] += dead_bias
    for first, end in getattr(match, "dead_spans", []):
        logp[first:end, :2] = -1e6
    return F.decode_halves(match, logp, switch_team, switch_dead)
