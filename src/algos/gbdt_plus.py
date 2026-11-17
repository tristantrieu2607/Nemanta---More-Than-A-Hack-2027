"""gbdt_viterbi with switchable feature groups and a minimum-possession post-process.

Feature groups (fit parameter `groups`, comma-separated):
  base      src/algos/_features.py                         (= gbdt_viterbi)
  rule      control_tracker state                           (= gbdt_hybrid)
  setpiece  restart layouts from player positions           (src/algos/_setpiece.py)
  asym      past-only vs future-only context (change points: control before vs after a frame)
  shape     past vs future team shape per team, in each team's own attacking frame: depth, width,
            length, players near the ball, speed (a team that wins the ball spreads and pushes up)
  shape2    more of the same: defensive-line height (2nd-deepest player, skips the GK), team length,
            players goal-side of the ball, over 1 / 3 / 10 / 20 s before vs after

Decode: `dead_bias` (log units) is added to the dead-ball log-probability before Viterbi (>0 calls
more stoppages); Viterbi with switch costs, then possessions shorter than `min_spell_s` are absorbed by the
possession they interrupt unless they start right after a dead ball -- the same reading of "under
control" the benchmark labels use (labels.py). 0 = off.
"""
import numpy as np

from src.algos import _features as F
from src.algos import _learned
from src.algos import _setpiece as SP
from src.algos import gbdt_hybrid as GH
from src.algos import gbdt_viterbi as G

PARAMS = dict(G.PARAMS, switch_team=16.0, switch_dead=12.0, stride=16, groups="base,asym,shape,shape2",
              min_spell_s=0.0, dead_bias=0.0, train_reading="main")
_cache = {}


def _asym(X, names):
    col = {n: X[:, i] for i, n in enumerate(names)}
    near = col["near_ctrl"]
    out, nm = [], []
    for w in (25, 75, 250):
        past = _causal_mean(near, w)
        fut = _causal_mean(near[::-1], w)[::-1]
        out += [past, fut, fut - past]
        nm += ["past_ctrl_%d" % w, "fut_ctrl_%d" % w, "jump_ctrl_%d" % w]
    for w in (75, 250):
        sp = _causal_mean(col["v_all"], w)
        sf = _causal_mean(col["v_all"][::-1], w)[::-1]
        out += [sp, sf]
        nm += ["past_v_%d" % w, "fut_v_%d" % w]
    return np.stack(out, 1).astype(np.float32), nm


def _shape(X, names):
    col = {n: X[:, i] for i, n in enumerate(names)}
    own = {"h": {"cx": col["cx_h"]}, "a": {"cx": 1 - col["cx_a"]}}  # each team's own attack frame
    out, nm = [], []
    for t in ("h", "a"):
        series = {"cx": own[t]["cx"], "sx": col["sx_" + t], "sy": col["sy_" + t],
                  "n8": col["n8_" + t], "v": col["v_" + t]}
        for k, x in series.items():
            for w in (75, 250):
                past = _causal_mean(x, w)
                fut = _causal_mean(x[::-1], w)[::-1]
                out += [past, fut, fut - past]
                nm += ["shp_%s_p%d_%s" % (k, w, t), "shp_%s_f%d_%s" % (k, w, t),
                       "shp_%s_j%d_%s" % (k, w, t)]
    return np.stack(out, 1).astype(np.float32), nm


def _shape2(match):
    import warnings
    L = match.pitch[0]
    right = F.home_attacks_right(match)[:, None]
    x = match.players_xy[..., 0]
    ball = F.interpolate_ball(match.ball_xy, match.ball_seen)[:, 0]
    out, nm = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t, tn in ((0, "h"), (1, "a")):
            mine = match.track_team == t
            # own attacking frame: 0 = own goal, 1 = opponent goal
            own_right = right if t == 0 else ~right
            xo = np.where(own_right, x[:, mine] / L, 1 - x[:, mine] / L)
            bo = np.where(own_right[:, 0], ball / L, 1 - ball / L)
            srt = np.sort(np.where(np.isnan(xo), np.inf, xo), axis=1)
            line = np.where(np.isfinite(srt[:, 1]), srt[:, 1], np.nan)  # 2nd deepest: skip the GK
            q = np.nanpercentile(xo, [10, 90], axis=1)
            length = q[1] - q[0]
            goal_side = np.nansum(xo < bo[:, None], axis=1).astype(float)
            for k, series in (("line", line), ("len", length), ("goalside", goal_side)):
                for w in (25, 75, 250, 500):
                    past = _causal_mean(series, w)
                    fut = _causal_mean(series[::-1], w)[::-1]
                    out += [past, fut, fut - past]
                    nm += ["sh2_%s_p%d_%s" % (k, w, tn), "sh2_%s_f%d_%s" % (k, w, tn),
                           "sh2_%s_j%d_%s" % (k, w, tn)]
    return np.stack(out, 1).astype(np.float32), nm


def _causal_mean(x, w):
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    cnt = n[i] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (c[i] - c[lo]) / cnt
    m[cnt == 0] = np.nan
    return m


def featurizer(groups):
    groups = tuple(g.strip() for g in groups.split(","))

    def featurize(match):
        key = (groups, match.game)
        if key not in _cache:
            X, names = GH.featurize(match) if "rule" in groups else F.frame_features(match)
            parts, allnames = [X], list(names)
            if "setpiece" in groups:
                S, sn = SP.setpiece_features(match)
                parts.append(S)
                allnames += sn
            if "asym" in groups or "shape" in groups:
                base, bn = F.frame_features(match)
            if "asym" in groups:
                Aa, an = _asym(base, bn)
                parts.append(Aa)
                allnames += an
            if "shape" in groups:
                Sh, shn = _shape(base, bn)
                parts.append(Sh)
                allnames += shn
            if "shape2" in groups:
                S2, s2n = _shape2(match)
                parts.append(S2)
                allnames += s2n
            _cache.clear()
            _cache[key] = (np.concatenate(parts, 1), allnames)
        return _cache[key]

    return featurize


def mirror(X, names):
    Y = GH.mirror(X, names) if "rule_team" in names else F.mirror(X, names)
    idx = {n: i for i, n in enumerate(names)}
    for n, i in idx.items():
        if n.startswith("sp_") and n.endswith("_h"):
            j = idx[n[:-2] + "_a"]
            Y[:, i], Y[:, j] = X[:, j], X[:, i]
        if n.startswith(("past_ctrl", "fut_ctrl", "jump_ctrl")):
            Y[:, i] = -X[:, i]
    return Y


def absorb_short(labels, match, min_spell_s):
    if min_spell_s <= 0:
        return labels
    out = labels.copy()
    min_len = int(min_spell_s * F.FPS)
    code = np.select([out == "H", out == "A"], [0, 1], -1)
    for _, first, last in match.halves:
        seg = code[first:last + 1]
        for s, e in F.runs(seg >= 0):  # each live run
            run = seg[s:e]
            b = np.concatenate(([0], np.flatnonzero(np.diff(run)) + 1, [len(run)]))
            spells = [[b0, b1, run[b0]] for b0, b1 in zip(b[:-1], b[1:])]
            kept = [spells[0]]
            for sp in spells[1:]:
                if sp[1] - sp[0] < min_len or sp[2] == kept[-1][2]:
                    kept[-1][1] = sp[1]
                else:
                    kept.append(sp)
            for b0, b1, t in kept:
                out[first + s + b0:first + s + b1] = "H" if t == 0 else "A"
    return out


def fit(matches, groups=PARAMS["groups"], **params):
    tag = "gbdt_plus:" + groups
    return _learned.fit(tag, featurizer(groups), mirror, matches, **params)


def predict(match, model=None, groups=PARAMS["groups"], switch_team=16.0, switch_dead=12.0,
            min_spell_s=0.0, dead_bias=0.0, **_):
    if model is None:
        raise ValueError("needs a fitted model (src.run and the bench fit it)")
    P = _learned.probabilities(featurizer(groups), mirror, match, model)
    logp = np.log(P + 1e-6)
    logp[:, 2] += dead_bias
    labels = F.decode_halves(match, logp, switch_team, switch_dead)
    return absorb_short(labels, match, min_spell_s)
