"""The external benchmark: every decision is made here, never on BAR-ATH.

    python -m src.external.bench convert              # build the caches from data/external/ (once)
    python -m src.external.bench info                 # what is cached, label statistics per match
    python -m src.external.bench baseline -p window=25,50,125   # score an algo (no logging)

Normal use is through `python -m src.run <algo>`, which logs the benchmark next to the public check.

An algorithm is scored on every cached external match x degradation seed, with the organisers'
possession_metrics.evaluate. Reported per source (idsse, skillcorner, metrica) and as `balanced`,
the mean of the source means, so the 20 SkillCorner matches do not drown the 7 DFL ones.

Algorithms that learn: define `fit(matches, **params) -> model` next to `predict(match, model=None,
**params)`. Each match is then scored by a model fitted without it (grouped k-fold over matches).
"""
import argparse
import hashlib
import importlib
import sys
import time

import numpy as np

import possession_metrics as pm
from src.external import degrade as G
from src.external import labels as L
from src.external import schema

SOURCES = ("idsse", "skillcorner", "metrica")
DEFAULT_SEEDS = (0, 1)
DEFAULT_PROFILE = "atd"
DEFAULT_READINGS = ("main", "short")
FOLDS = 5


class ExternalMatch:
    """Looks like src.common.Match to an algorithm. `gt(reading)` gives the labels (never pass
    them to predict)."""

    def __init__(self, clean, profile=DEFAULT_PROFILE, seed=0):
        d = G.degrade(clean, profile, seed)
        self.game = "{}#{}".format(clean.name, seed)
        self.source, self.match_id, self.seed = clean.source, clean.match_id, seed
        self.pitch = clean.pitch
        self.n_frames = clean.n_frames
        self.players_xy = d["players_xy"]
        self.ball_xy = d["ball_xy"]
        self.ball_seen = ~np.isnan(self.ball_xy[:, 0])
        self.players_speed = _speed(self.players_xy)  # m/s from positions (ATD speed unit unverified)
        self.ball_speed = _speed(self.ball_xy[:, None, :])[:, 0]
        self.track_team = d["track_team"]
        self.track_is_gk = d["track_is_gk"]
        self.graded = clean.graded
        self.halves = list(clean.halves)
        self.dead_spans = []  # documented pre-kickoff waits exist only for BAR-ATH
        self.tags = []  # no Smart Tagging off BAR-ATH: tag-based logic cannot be validated here
        self._clean = clean

    def blank(self, fill="X"):
        return np.full(self.n_frames, fill, dtype="<U1")

    def gt(self, reading="main"):
        c = self._clean
        return L.challenge_codes(c.alive, c.owner, c.graded, L.READINGS[reading])


def _speed(xy, k=2):
    v = np.full(xy.shape[:2], np.nan, np.float32)
    v[k:-k] = np.hypot(*(xy[2 * k:] - xy[:-2 * k]).transpose(2, 0, 1)) * (G.FPS / (2 * k))
    return v


class LazyMatches:
    """The training set handed to `fit`: builds each degraded ExternalMatch only when iterated,
    so a fold never holds dozens of matches in memory. `names` identifies the set (for caches)."""

    def __init__(self, cleans, keys, seeds, profile):
        self._items = [(cleans[k], s) for k in keys for s in seeds]
        self._profile = profile
        self.names = tuple("{}#{}#{}".format(c.name, s, profile) for c, s in self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        for clean, seed in self._items:
            yield ExternalMatch(clean, self._profile, seed)


def matches(sources=SOURCES):
    return [(s, m) for s in sources for m in schema.cached(s)]


def _folds(keys, k=FOLDS):
    """Deterministic grouped folds over matches (md5, not hash(): hash() changes every process)."""
    order = sorted(keys, key=lambda key: hashlib.md5(":".join(key).encode()).hexdigest())
    return [order[i::k] for i in range(min(k, len(order)))]


def quick_subset(keys, every=3):
    """Screening subset: every 3rd match of the big sources, all of the small ones (<= 7 matches)."""
    out = []
    for source in SOURCES:
        ks = [k for k in keys if k[0] == source]
        out += ks if len(ks) <= 7 else ks[::every]
    return out


def evaluate(module, params, sources=SOURCES, seeds=DEFAULT_SEEDS, profile=DEFAULT_PROFILE,
             readings=DEFAULT_READINGS, bootstrap=0, verbose=False, quick=False):
    keys = matches(sources)
    if quick:
        keys = quick_subset(keys)
    if not keys:
        return None
    cleans = {key: schema.load(*key) for key in keys}
    models = {}
    if hasattr(module, "fit"):
        for test in _folds(keys):
            train = LazyMatches(cleans, [k for k in keys if k not in test], seeds, profile)
            model = module.fit(train, **params)
            models.update({k: model for k in test})
    rows = []
    for key in keys:
        for seed in seeds:
            match = ExternalMatch(cleans[key], profile, seed)
            started = time.time()
            extra = {"model": models[key]} if key in models else {}
            labels = np.asarray(module.predict(match, **dict(params, **extra)))
            codes = pm.label_codes(labels)
            for reading in readings:
                result = pm.evaluate(match.gt(reading), codes, bootstrap_n=bootstrap)
                rows.append({"source": key[0], "match": key[1], "seed": seed, "reading": reading,
                             "spell": round(float(result["spell_f1_mean"]), 4),
                             "macro": round(float(result["macro_f1"]), 4),
                             "n_pred": int(result["n_pred_spells"]), "n_gt": int(result["n_gt_spells"]),
                             "team_acc": round(float(result["team_acc_gt_alive"] or 0), 4),
                             "alive_dead": round(float(result["alive_dead_f1"] or 0), 4)})
            if verbose:
                r = rows[-len(readings)]
                print("  {:<22} seed {}  spellF1 {:.3f}  poss {}/{}  ({:.1f}s)".format(
                    "{}:{}".format(*key), seed, r["spell"], r["n_pred"], r["n_gt"], time.time() - started))
    return {"profile": profile, "seeds": list(seeds), "readings": list(readings),
            "summary": summarise(rows), "per_match": rows}


def summarise(rows):
    out = {}
    for reading in sorted({r["reading"] for r in rows}):
        rr = [r for r in rows if r["reading"] == reading]
        by_source = {}
        for source in SOURCES:
            sr = [r for r in rr if r["source"] == source]
            if not sr:
                continue
            per_match = {}
            for r in sr:
                per_match.setdefault(r["match"], []).append(r)
            match_spell = [np.mean([r["spell"] for r in v]) for v in per_match.values()]
            seed_means = [np.mean([r["spell"] for r in sr if r["seed"] == s]) for s in sorted({r["seed"] for r in sr})]
            by_source[source] = {
                "spell": round(float(np.mean(match_spell)), 4),
                "spell_min": round(float(np.min(match_spell)), 4),
                "spell_seed_sd": round(float(np.std(seed_means)), 4),
                "macro": round(float(np.mean([r["macro"] for r in sr])), 4),
                "poss_ratio": round(float(np.sum([r["n_pred"] for r in sr]) / max(1, np.sum([r["n_gt"] for r in sr]))), 3),
                "n_matches": len(per_match)}
        out[reading] = {"sources": by_source,
                        "balanced_spell": round(float(np.mean([v["spell"] for v in by_source.values()])), 4),
                        "balanced_macro": round(float(np.mean([v["macro"] for v in by_source.values()])), 4)}
    return out


def format_summary(result, indent="  "):
    if result is None:
        return indent + "external      -- nothing cached: python -m src.external.fetch && python -m src.external.bench convert"
    lines = []
    for reading, s in result["summary"].items():
        parts = ["{} {:.3f} (min {:.3f}, sd {:.3f}, n {})".format(k, v["spell"], v["spell_min"], v["spell_seed_sd"], v["n_matches"])
                 for k, v in s["sources"].items()]
        lines.append("{}external/{:<6} balanced spellF1 {:.4f}  macroF1 {:.4f}  | {}".format(
            indent, reading, s["balanced_spell"], s["balanced_macro"], "  ".join(parts)))
    return "\n".join(lines)


# -- CLI -------------------------------------------------------------------------------------

def convert(sources=SOURCES, force=False):
    from src.external import idsse, metrica, skillcorner
    modules = {"idsse": idsse, "skillcorner": skillcorner, "metrica": metrica}
    for source in sources:
        module = modules[source]
        found = module.available()
        if not found:
            print("{:<12} no raw files in data/external/{}/ -- see validation/README.md".format(source, source))
        for match_id in found:
            if not force and match_id in schema.cached(source):
                continue
            started = time.time()
            clean = module.convert(match_id)
            clean.save()
            print("{:<12} {:<10} {} frames, {:.0f}s".format(source, match_id, clean.n_frames, time.time() - started))


def info(sources=SOURCES):
    print("{:<24} {:>7} {:>6} {:>6} {:>6} {:>8} {:>7}".format(
        "match", "min", "alive", "ball", "poss", "mean s", "poss1s"))
    for key in matches(sources):
        c = schema.load(*key)
        g = c.graded
        main = L.challenge_codes(c.alive, c.owner, g, L.READINGS["main"])
        short = L.challenge_codes(c.alive, c.owner, g, L.READINGS["short"])
        st = pm.possession_stats(main, pm.valid_runs(main))
        n_short = len(pm.possession_spells(short, pm.valid_runs(short)))
        print("{:<24} {:>7.1f} {:>6.2f} {:>6.2f} {:>6} {:>8.1f} {:>7}".format(
            c.name, g.sum() / 25 / 60, c.alive[g].mean(), (~np.isnan(c.ball_xy[g, 0])).mean(),
            st["n_spells"], st["mean_duration_s"] or 0, n_short))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", help="convert | info | <algo name in src/algos>")
    parser.add_argument("-p", "--param", action="append", default=[], metavar="KEY=V1[,V2]")
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(G.PROFILES))
    parser.add_argument("--readings", default=",".join(DEFAULT_READINGS))
    parser.add_argument("--force", action="store_true", help="convert: rebuild existing caches")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    sources = tuple(s for s in args.sources.split(",") if s)
    if args.what == "convert":
        return convert(sources, args.force)
    if args.what == "info":
        return info(sources)
    from src.run import parse_grid
    import itertools
    module = importlib.import_module("src.algos." + args.what)
    defaults = dict(getattr(module, "PARAMS", {}))
    grid = parse_grid(args.param)
    keys = sorted(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(defaults, **dict(zip(keys, values)))
        result = evaluate(module, params, sources, tuple(int(s) for s in args.seeds.split(",")),
                          args.profile, tuple(args.readings.split(",")), verbose=args.verbose)
        print("{} {}  [{} profile, seeds {}]".format(args.what, params, args.profile, args.seeds))
        print(format_summary(result))


if __name__ == "__main__":
    sys.exit(main())
