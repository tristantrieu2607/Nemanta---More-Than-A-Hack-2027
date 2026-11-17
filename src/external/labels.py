"""Turn provider possession flags into the challenge's possession rule.

The challenge ground truth (CHALLENGE.md): ball dead -> D; ball alive -> exactly one team has
possession; it changes only when the other team has the ball *under control* (a touch, a deflection
or a won header without control is not a change); every restart has a clear owner.

Provider flags are closer to "last touch": DFL's BallPossession flips on every touch (~500 spells a
match, median ~3 s). We bring every source to the challenge rule the same way:

  1. inside a live run, frames with no owner keep the last owner (loose ball stays with the team
     that had it); a live run that starts ownerless takes the first owner it gets;
  2. a spell shorter than `min_control_s` is absorbed by the possession it interrupts, unless it is
     the first spell after a restart (the restarting team is the clear possessor).

`min_control_s` is a reading of "under control", not a tuned number. Score under more than one
reading (READINGS) -- the private ground truth's exact convention is unknown, and the top public
teams see spellF1 move by >0.1 between readings.
"""
import numpy as np

import possession_metrics as pm

FPS = 25
H, A, D, X = (pm.LABELS.index(c) for c in "HADX")
READINGS = {"main": 2.0, "short": 1.0, "raw": 0.0}  # name -> min_control_s


def runs(mask):
    """(start, end_exclusive) of each run of True in a bool array."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_owner(owner, alive):
    """Owner (0 home, 1 away, -1 unknown) forward-filled inside each live run, then back-filled."""
    owner = np.where(alive, owner, -1).astype(np.int8)
    out = owner.copy()
    for start, end in runs(alive):
        seg = out[start:end]
        known = np.flatnonzero(seg >= 0)
        if not len(known):
            continue
        idx = np.maximum.accumulate(np.where(seg >= 0, np.arange(len(seg)), -1))
        seg[idx >= 0] = seg[idx[idx >= 0]]
        seg[:known[0]] = seg[known[0]]
    return out


def challenge_codes(alive, owner, graded, min_control_s=2.0, fps=FPS):
    """Per-frame H/A/D/X codes under the challenge rule. X outside `graded`."""
    alive = np.asarray(alive, dtype=bool)
    owner = fill_owner(owner, alive)
    min_len = int(round(min_control_s * fps))
    team = owner.copy()
    for start, end in runs(alive):
        seg = team[start:end]
        if (seg < 0).all():
            continue  # live run with no owner at all: leave unknown, becomes X below
        bounds = np.flatnonzero(np.diff(seg)) + 1
        spells = list(zip(np.concatenate(([0], bounds)), np.concatenate((bounds, [len(seg)]))))
        kept = [list(spells[0]) + [seg[spells[0][0]]]]
        for s, e in spells[1:]:
            t = seg[s]
            if (e - s) < min_len or t == kept[-1][2]:
                kept[-1][1] = e  # absorbed by (or continuing) the possession it interrupts
            else:
                kept.append([s, e, t])
        for s, e, t in kept:
            seg[s:e] = t
    codes = np.full(len(alive), D, dtype=np.int8)
    codes[alive & (team == 0)] = H
    codes[alive & (team == 1)] = A
    codes[alive & (team < 0)] = X
    codes[~np.asarray(graded, dtype=bool)] = X
    return codes
