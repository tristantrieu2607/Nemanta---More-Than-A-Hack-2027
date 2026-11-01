"""Run an algorithm (or a small sweep), write its submission, score it, log the run.

    python -m src.run baseline
    python -m src.run baseline -p window=25,50,125          # sweep: one run per value
    python -m src.run my_idea -p radius_m=1.5 --note "loose ball keeps last team"
    python -m src.run my_idea --bootstrap 1000              # intervals, before believing any gap
    python -m src.run my_idea --no-log                      # quick look, nothing recorded

Each logged run is one small JSON in experiments/runs/ -- commit it. Rank everything with
    python -m src.leaderboard
"""
import argparse
import ast
import datetime
import getpass
import importlib
import itertools
import json
import os
import re
import subprocess
import time

import numpy as np

from src import common

RUNS_DIR = os.path.join(common.ROOT, "experiments", "runs")


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


def line(name, scores):
    if scores is None:
        return "  {:<13} -- no hand labels yet (validation/labels/)".format(name)
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
    parser.add_argument("--bootstrap", type=int, default=0, help="draws for intervals, e.g. 1000")
    parser.add_argument("--no-log", action="store_true", help="do not record in experiments/runs")
    args = parser.parse_args()

    module = importlib.import_module("src.algos." + args.algo)
    defaults = dict(getattr(module, "PARAMS", {}))
    grid = parse_grid(args.param)
    unknown = sorted(set(grid) - set(defaults))
    if unknown:
        parser.error("{} has no params {} -- declare them in PARAMS".format(args.algo, unknown))

    match = common.Match()
    public = common.public_gt(match.game)
    val, val_files, val_hash = common.load_validation(match.n_frames)
    author = git("config", "user.name") or getpass.getuser()
    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", "src"))

    keys = sorted(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(defaults, **dict(zip(keys, values)))
        started = time.time()
        labels = np.asarray(module.predict(match, **params))
        runtime = time.time() - started
        name = args.algo + ("__" + tag(params) if params else "")
        path = common.write_submission(os.path.join(common.SUBMISSION_DIR, name + ".csv"), labels)
        codes = common.check_file(path, match.n_frames)

        record = {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "author": author, "commit": commit, "dirty": dirty,
            "algo": args.algo, "params": params, "note": args.note,
            "runtime_s": round(runtime, 1),
            "public": common.score(codes, public, args.bootstrap),
            "val": common.score(codes, val, args.bootstrap) if val is not None else None,
            "val_hash": val_hash,
            "val_files": [os.path.basename(f) for f in val_files],
        }
        print("{}  {}  ({:.1f}s){}".format(args.algo, params, runtime,
                                         "  [uncommitted src/]" if dirty else ""))
        print(line("public 5 min", record["public"]))
        print(line("validation", record["val"]))
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
