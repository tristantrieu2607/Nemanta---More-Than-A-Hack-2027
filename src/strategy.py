"""Score one strategy on the external benchmark and add it to strategies/BOARD.md.

    python -m src.strategy <algo> --title "what the idea is, in plain words" [--quick] [-p k=v]

--quick  screening run: 10 matches (7 SkillCorner + 3 Metrica; + DFL once cached), 1 noise seed.
         Minutes instead of the best part of an hour for learned algos. Quick scores are only
         comparable with other quick scores (learned algos train on fewer matches).
full     every cached match, 2 noise seeds. Run it for anything that wins a quick run.

The board is the plain-language scoreboard for the team; experiments/runs/ keeps the full JSON.
Nothing here touches BAR-ATH: use `python -m src.run <algo>` to write a submission.
"""
import argparse
import datetime
import importlib
import json
import os
import re
import time

from src import common
from src.external import bench
from src.run import RUNS_DIR, git, parse_grid, tag

BOARD = os.path.join(common.ROOT, "strategies", "BOARD.md")
HEADER = """# Strategy board

Scored on other matches made to look like ours (never on Barça–Athletic). Higher is better.
`score` = balanced spellF1, main labelling. `loose` = same with the looser 1 s labelling: a real
improvement goes up on both. `worst` = lowest single match. Differences under 0.01 are noise.
Quick and full runs are separate leagues: only compare rows of the same mode.

| date | strategy | algo file | mode | score | loose | worst | vs best before | verdict |
|---|---|---|---|---|---|---|---|---|
"""


def best_before(mode):
    best = None
    if os.path.isfile(BOARD):
        for line in open(BOARD, encoding="utf-8"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 9 and cells[3] == mode:
                try:
                    best = max(best or 0.0, float(cells[4].strip("*")))
                except ValueError:
                    pass
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("algo")
    parser.add_argument("--title", required=True, help="the idea in plain words")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--verdict", default="", help="kept / dropped + one line why (edit later)")
    args = parser.parse_args()

    module = importlib.import_module("src.algos." + args.algo)
    params = dict(getattr(module, "PARAMS", {}))
    params.update({k: v[0] for k, v in parse_grid(args.param).items()})
    mode = "quick" if args.quick else "full"
    seeds = (0,) if args.quick else bench.DEFAULT_SEEDS
    started = time.time()
    result = bench.evaluate(module, params, seeds=seeds, quick=args.quick)
    if result is None:
        raise SystemExit("no external matches cached -- see validation/README.md")
    main_, loose = result["summary"]["main"], result["summary"]["short"]
    worst = min(v["spell_min"] for v in main_["sources"].values())
    before = best_before(mode)
    delta = "first" if before is None else "{:+.3f}".format(main_["balanced_spell"] - before)
    print(bench.format_summary(result))
    print("{} run, {:.0f} min. vs best {} before: {}".format(mode, (time.time() - started) / 60, mode, delta))

    os.makedirs(os.path.dirname(BOARD), exist_ok=True)
    if not os.path.isfile(BOARD):
        with open(BOARD, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(HEADER)
    changed = {k: v for k, v in params.items() if getattr(module, "PARAMS", {}).get(k) != v}
    name = args.algo + (" (" + ", ".join("{}={}".format(k, v) for k, v in changed.items()) + ")" if changed else "")
    with open(BOARD, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("| {} | {} | `{}` | {} | {:.3f} | {:.3f} | {:.3f} | {} | {} |\n".format(
            datetime.date.today().isoformat(), args.title.replace("|", "/"), name, mode,
            main_["balanced_spell"], loose["balanced_spell"], worst, delta, args.verdict))

    os.makedirs(RUNS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    record = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "algo": args.algo,
              "params": params, "note": args.title, "mode": mode, "commit": git("rev-parse", "--short", "HEAD"),
              "external": result, "public": None, "bench": "strategy-" + mode,
              "author": git("config", "user.name") or "strategy", "dirty": True}
    out = os.path.join(RUNS_DIR, "{}_strategy-{}_{}.json".format(
        stamp, mode, re.sub(r"[^A-Za-z0-9._-]", "", args.algo + "__" + tag(changed))[:90]))
    with open(out, "w") as handle:
        json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
