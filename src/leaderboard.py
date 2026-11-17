"""Rank every logged run in experiments/runs/ on the external benchmark.

    python -m src.leaderboard                 # latest run per algo+params, ranked
    python -m src.leaderboard --algo baseline # one algorithm only
    python -m src.leaderboard --all           # every run, repeats and stale ones included

Ranked on balanced external spellF1 under the main label reading (mean of the DFL, SkillCorner and
Metrica means). `short` is the same benchmark under a looser reading of "under control" -- a change
that wins on main but loses on short is fitting a labelling convention, not football. The public 5
minutes are shown as a pipeline check and never used to rank.
"""
import argparse
import glob
import json
import os

import pandas as pd

from src.external import bench
from src.run import RUNS_DIR, bench_signature


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algo")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        with open(path) as handle:
            runs.append(json.load(handle))
    runs = [r for r in runs if not args.algo or r["algo"] == args.algo]
    if not runs:
        print("no runs logged yet -- python -m src.run baseline")
        return

    current = bench_signature(bench.SOURCES, bench.DEFAULT_SEEDS, bench.DEFAULT_PROFILE,
                              bench.DEFAULT_READINGS)
    rows = []
    for r in runs:
        ext = (r.get("external") or {}).get("summary", {})
        main_, short = ext.get("main", {}), ext.get("short", {})
        src = main_.get("sources", {})
        pub = r.get("public") or {}
        rows.append({
            "algo": r["algo"],
            "params": " ".join("{}={}".format(k, v) for k, v in sorted(r["params"].items())),
            "ext_spell": main_.get("balanced_spell"),
            "ext_short": short.get("balanced_spell"),
            "ext_macro": main_.get("balanced_macro"),
            "idsse": src.get("idsse", {}).get("spell"),
            "skillc": src.get("skillcorner", {}).get("spell"),
            "metrica": src.get("metrica", {}).get("spell"),
            "worst": min([v["spell_min"] for v in src.values()], default=None),
            "pub_check": pub.get("spell_f1_mean"),
            "who": r["author"],
            "commit": r["commit"] + ("*" if r["dirty"] else ""),
            "time": r["time"],
            "note": r.get("note", ""),
            "_stale": r.get("bench") != current,
        })
    table = pd.DataFrame(rows)
    hidden = 0
    if not args.all:
        table = table.sort_values("time").drop_duplicates(["algo", "params"], keep="last")
        hidden = int(table["_stale"].sum())
        table = table[~table["_stale"]]
    table = table.sort_values("ext_spell", ascending=False, na_position="last").head(args.top)
    table.insert(0, "rank", range(1, len(table) + 1))
    print("Ranked on external balanced spellF1 (benchmark {}). Differences under ~0.01 are seed noise;"
          " pub_check is a pipeline check only.".format(current))
    if hidden:
        print("{} run(s) hidden: scored on another benchmark (data, seeds or bench version) -- rerun."
              .format(hidden))
    print("* = run from uncommitted code\n")
    with pd.option_context("display.width", 220, "display.max_colwidth", 36):
        print(table.drop(columns=["_stale"]).to_string(index=False))


if __name__ == "__main__":
    main()
