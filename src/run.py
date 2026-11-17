"""Run an algorithm (or a small sweep): external benchmark, BAR-ATH submission, public check, log.

    python -m src.run baseline
    python -m src.run baseline -p window=25,50,125          # sweep: one run per value
    python -m src.run my_idea -p radius_m=1.5 --note "loose ball keeps last team"
    python -m src.run my_idea --external-only               # no BAR-ATH needed (no submission)
    python -m src.run my_idea --no-log                      # quick look, nothing recorded

Decisions are made on the EXTERNAL benchmark (src/external/bench.py: DFL, SkillCorner, Metrica,
degraded to look like the ATD). The organisers' 3 Nov rules: tune only on external data, no hand
labels on BAR-ATH. The public 5 minutes are printed as a pipeline check -- never choose on them.

Each logged run is one small JSON in experiments/runs/ -- commit it. Rank with
    python -m src.leaderboard
"""
import argparse
import ast
import datetime
import getpass
import hashlib
import importlib
import itertools
import json
import os
import re
import subprocess
import time

import numpy as np

from src import common
from src.external import bench, schema
from src.external.degrade import PROFILES

RUNS_DIR = os.path.join(common.ROOT, "experiments", "runs")
BENCH_VERSION = 1  # bump when labels.py / degrade.py change meaning, so old runs show as stale


def literal(text):
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return text.strip()


def parse_grid(pairs):
    grid = {}
    for pair in pairs:
        key, _, values = pair.partition("=")
        grid[key.strip()] = [literal(v) for v in values.split(",")]
    return grid


def git(*args):
    try:
        out = subprocess.run(["git", "--no-optional-locks", *args], cwd=common.ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def tag(params):
    text = "_".join("{}-{}".format(k, params[k]) for k in sorted(params))
    return re.sub(r"[^A-Za-z0-9._-]", "", text)[:80]


def bench_signature(sources, seeds, profile, readings):
    """Runs are comparable only on the same benchmark: same matches, seeds, profile, readings."""
    keys = [":".join(k) for k in bench.matches(sources)]
    text = json.dumps([BENCH_VERSION, keys, list(seeds), profile, list(readings)])
    return hashlib.md5(text.encode()).hexdigest()[:8]


def line(name, scores):
    ci = ""
    if scores["spell_bootstrap_low"] is not None:
        ci = " [{:.3f}-{:.3f}]".format(scores["spell_bootstrap_low"], scores["spell_bootstrap_high"])
    return "  {:<13} spellF1 {:.4f}{}  macroF1 {:.4f}  poss {}/{}  mean {:.1f}s/{:.1f}s".format(
        name, scores["spell_f1_mean"], ci, scores["macro_f1"], scores["n_pred_spells"],
        scores["n_gt_spells"], scores["mean_spell_duration_s"] or 0,
        scores["gt_mean_spell_duration_s"] or 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("algo", help="file name in src/algos/, without .py")
    parser.add_argument("-p", "--param", action="append", default=[], metavar="KEY=V1[,V2]")
    parser.add_argument("--note", default="", help="what this run tests, in a few words")
    parser.add_argument("--bootstrap", type=int, default=0, help="public-check intervals, e.g. 1000")
    parser.add_argument("--sources", default=",".join(bench.SOURCES))
    parser.add_argument("--seeds", default=",".join(map(str, bench.DEFAULT_SEEDS)))
    parser.add_argument("--profile", default=bench.DEFAULT_PROFILE, choices=sorted(PROFILES))
    parser.add_argument("--readings", default=",".join(bench.DEFAULT_READINGS))
    parser.add_argument("--external-only", action="store_true", help="skip BAR-ATH (no data needed)")
    parser.add_argument("--no-external", action="store_true", help="skip the benchmark (not rankable)")
    parser.add_argument("--no-log", action="store_true", help="do not record in experiments/runs")
    args = parser.parse_args()

    module = importlib.import_module("src.algos." + args.algo)
    defaults = dict(getattr(module, "PARAMS", {}))
    grid = parse_grid(args.param)
    unknown = sorted(set(grid) - set(defaults))
    if unknown:
        parser.error("{} has no params {} -- declare them in PARAMS".format(args.algo, unknown))
    sources = tuple(s for s in args.sources.split(",") if s)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    readings = tuple(args.readings.split(","))
    signature = bench_signature(sources, seeds, args.profile, readings)

    match = None if args.external_only else common.Match()
    public = None if match is None else common.public_gt(match.game)
    author = git("config", "user.name") or getpass.getuser()
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "src"))

    keys = sorted(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(defaults, **dict(zip(keys, values)))
        name = args.algo + ("__" + tag(params) if params else "")
        record = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "author": author, "commit": commit, "dirty": dirty,
            "algo": args.algo, "params": params, "note": args.note,
            "public": None, "external": None, "bench": signature,
        }
        print("{}  {}{}".format(args.algo, params, "  [uncommitted src/]" if dirty else ""))
        if not args.no_external:
            started = time.time()
            record["external"] = bench.evaluate(module, params, sources, seeds, args.profile, readings)
            record["external_runtime_s"] = round(time.time() - started, 1)
            print(bench.format_summary(record["external"]))
        if match is not None:
            if hasattr(module, "fit"):
                keys_all = bench.matches(sources)
                train = bench.LazyMatches({k: schema.load(*k) for k in keys_all}, keys_all, seeds,
                                          args.profile)
                params_bar = dict(params, model=module.fit(train, **params))
            else:
                params_bar = params
            started = time.time()
            labels = np.asarray(module.predict(match, **params_bar))
            record["runtime_s"] = round(time.time() - started, 1)
            path = common.write_submission(os.path.join(common.SUBMISSION_DIR, name + ".csv"), labels)
            codes = common.check_file(path, match.n_frames)
            record["public"] = common.score(codes, public, args.bootstrap)
            print(line("public 5 min", record["public"]) + "   <- pipeline check only")
            print("  -> " + os.path.relpath(path, common.ROOT))
        if not args.no_log:
            os.makedirs(RUNS_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            who = re.sub(r"[^A-Za-z0-9]", "", author)[:12] or "anon"
            out = os.path.join(RUNS_DIR, "{}_{}_{}.json".format(stamp, who, name[:90]))
            with open(out, "w") as handle:
                json.dump(record, handle, indent=1)
            print("  logged " + os.path.relpath(out, common.ROOT))


if __name__ == "__main__":
    main()
