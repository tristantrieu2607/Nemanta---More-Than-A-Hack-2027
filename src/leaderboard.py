"""Rank every logged run in experiments/runs/.

    python -m src.leaderboard                 # latest run per algo+params, ranked
    python -m src.leaderboard --algo baseline # one algorithm only
    python -m src.leaderboard --all           # every run, repeats and stale ones included

Ranked on validation spellF1 once hand labels exist. Until then it ranks on the public 5 minutes,
which is a sanity check, not a ranking: 17 possessions, ~6 points of noise per possession.
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
    parser.add_argument("--top", type=int, default=30)
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

    _, _, current_hash = common.load_validation(common.source_frames("tactical"))
    rows = []
    for r in runs:
        val = r.get("val") if r.get("val_hash") == current_hash else None
        stale = r.get("val") is not None and r.get("val_hash") != current_hash
        rows.append({
            "algo": r["algo"],
            "params": " ".join("{}={}".format(k, v) for k, v in sorted(r["params"].items())),
            "val_spell": val["spell_f1_mean"] if val else None,
            "val_ci": ("{:.3f}-{:.3f}".format(val["spell_bootstrap_low"], val["spell_bootstrap_high"])
                       if val and val["spell_bootstrap_low"] is not None else ""),
            "val_macro": val["macro_f1"] if val else None,
            "val_poss": "{}/{}".format(val["n_pred_spells"], val["n_gt_spells"]) if val else "",
            "pub_spell": r["public"]["spell_f1_mean"],
            "pub_macro": r["public"]["macro_f1"],
            "pub_poss": "{}/{}".format(r["public"]["n_pred_spells"], r["public"]["n_gt_spells"]),
            "who": r["author"],
            "commit": r["commit"] + ("*" if r["dirty"] else ""),
            "time": r["time"],
            "note": r.get("note", ""),
            "_stale": stale,
        })
    table = pd.DataFrame(rows)
    if not args.all:
        table = table.sort_values("time").drop_duplicates(["algo", "params"], keep="last")
        hidden = int(table["_stale"].sum())
        table = table[~table["_stale"]]
    else:
        hidden = 0

    has_val = table["val_spell"].notna().any()
    key = "val_spell" if has_val else "pub_spell"
    table = table.sort_values(key, ascending=False, na_position="last").head(args.top)
    table.insert(0, "rank", range(1, len(table) + 1))
    if not has_val:
        table = table.drop(columns=["val_spell", "val_ci", "val_macro", "val_poss"])
        print("Ranked on the PUBLIC 5 minutes -- no hand labels yet. Sanity check only, not a ranking.")
    else:
        print("Ranked on validation spellF1 (labels {}). Gaps under ~0.04 are noise; check val_ci."
              .format(current_hash))
    if hidden:
        print("{} run(s) hidden: scored on an older validation set -- rerun them.".format(hidden))
    print("* = run from uncommitted code\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(table.drop(columns=["_stale"]).to_string(index=False))


if __name__ == "__main__":
    main()
