"""Every logged run in experiments/runs/, side by side.

    python -m src.leaderboard                 # latest run per algo+params
    python -m src.leaderboard --algo baseline # one algorithm only
    python -m src.leaderboard --all           # every run, repeats included

Ranked on validation spellF1 once a validation set exists (other games, see validation/README.md).
Until then this is a list, NOT a ranking: the public 5 minutes are a pipeline check, and RULES.md
forbids using them to choose between versions.
"""
import argparse
import glob
import json
import os

import pandas as pd

from src import common
from src.run import RUNS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algo")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        with open(path) as handle:
            runs.append(json.load(handle))
    if args.algo:
        runs = [r for r in runs if r["algo"] == args.algo]
    if not runs:
        print("no runs logged yet -- python -m src.run baseline")
        return

    _, _, current_hash = common.load_validation()
    rows = []
    for r in runs:
        current = r.get("val") is not None and r.get("val_hash") == current_hash
        val = r["val"] if current else None
        rows.append({
            "algo": r["algo"],
            "params": " ".join("{}={}".format(k, v) for k, v in sorted(r["params"].items())),
            "val_spell": val["spell_f1_mean"] if val else None,
            "val_ci": ("{:.3f}-{:.3f}".format(val["spell_bootstrap_low"], val["spell_bootstrap_high"])
                       if val and val["spell_bootstrap_low"] is not None else ""),
            "val_macro": val["macro_f1"] if val else None,
            "val_poss": "{}/{}".format(val["n_pred_spells"], val["n_gt_spells"]) if val else "",
            "check_spell": r["public"]["spell_f1_mean"],
            "check_poss": "{}/{}".format(r["public"]["n_pred_spells"], r["public"]["n_gt_spells"]),
            "who": r["author"],
            "commit": r["commit"] + ("*" if r["dirty"] else ""),
            "time": r["time"],
            "note": r.get("note", ""),
            "_stale": r.get("val") is not None and not current,
        })
    table = pd.DataFrame(rows)
    hidden = 0
    if not args.all:
        table = table.sort_values("time").drop_duplicates(["algo", "params"], keep="last")
        hidden = int(table["_stale"].sum())
        table = table[~table["_stale"]]

    if table["val_spell"].notna().any():
        table = table.sort_values("val_spell", ascending=False, na_position="last")
        table.insert(0, "rank", range(1, len(table) + 1))
        print("Ranked on validation spellF1 (set {}). Gaps inside val_ci are noise.".format(
            current_hash))
    else:
        table = table.drop(columns=["val_spell", "val_ci", "val_macro", "val_poss"])
        table = table.sort_values("time", ascending=False)
        print("NOT A RANKING -- no validation set yet (needs other games, validation/README.md).")
        print("check_* = public 5 min: confirms the pipeline works and the possession count is sane.")
        print("Choosing between versions on it is against RULES.md.")
    if hidden:
        print("{} run(s) hidden: scored on an older validation set -- rerun them.".format(hidden))
    print("* = run from uncommitted code\n")
    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        print(table.drop(columns=["_stale"]).head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
