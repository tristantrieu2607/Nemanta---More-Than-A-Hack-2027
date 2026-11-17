"""gbdt_plus + the documented wait before the second-half kickoff is dead ball.

The organisers' FAQ states that the second half is graded from frame 110,808 but does not kick off
until 113,742, and that those 2,934 graded frames are all `D`. A model that has never seen a
two-minute wait with players walking to their positions tends to call part of it live (Barca-
Hackathon measured 63% on their network before adding the same rule). `match.dead_spans` carries
that stretch (src/common.py, from the games.py kickoff anchors); inside it H and A are forbidden
before decoding, so the Viterbi path also starts the half cleanly at the kickoff.

External matches have no such stretch, so benchmark scores are identical to gbdt_plus by
construction; the rule is a documented fact about the file layout, not a tuned parameter.
"""
import numpy as np

from src.algos import _features as F
from src.algos import _learned
from src.algos import gbdt_plus as GP

PARAMS = dict(GP.PARAMS)


def fit(matches, **params):
    return GP.fit(matches, **params)


def predict(match, model=None, groups=PARAMS["groups"], switch_team=16.0, switch_dead=12.0,
            min_spell_s=0.0, dead_bias=0.0, **_):
    if model is None:
        raise ValueError("needs a fitted model (src.run and the bench fit it)")
    P = _learned.probabilities(GP.featurizer(groups), GP.mirror, match, model)
    logp = np.log(P + 1e-6)
    logp[:, 2] += dead_bias
    for first, end in getattr(match, "dead_spans", []):
        logp[first:end, :2] = -1e6
    labels = F.decode_halves(match, logp, switch_team, switch_dead)
    return GP.absorb_short(labels, match, min_spell_s)
