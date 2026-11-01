"""Upload an idea: commit its algo file and run logs, pull teammates' work, push.

    python -m src.share my_idea
    python -m src.share my_idea -m "my_idea: switch only within 1.5 m"

Commits src/algos/<idea>.py, experiments/, EXPERIMENTS.md and BUILD_NOTES.md -- nothing else,
so data and scratch files can never be uploaded by accident.
"""
import argparse
import os
import subprocess
import sys

from src import common

SHARED = ["experiments", "EXPERIMENTS.md", "BUILD_NOTES.md"]


def git(*args, check=True):
    print("> git " + " ".join(args))
    result = subprocess.run(["git", *args], cwd=common.ROOT)
    if check and result.returncode:
        sys.exit("git {} failed -- fix the message above, then rerun python -m src.share".format(
            args[0]))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("algo", help="file name in src/algos/, without .py")
    parser.add_argument("-m", "--message", help="commit message (default: '<algo>: new runs')")
    args = parser.parse_args()

    algo_file = os.path.join("src", "algos", args.algo + ".py")
    if not os.path.isfile(os.path.join(common.ROOT, algo_file)):
        parser.error("{} not found".format(algo_file))

    paths = [algo_file] + [p for p in SHARED if os.path.exists(os.path.join(common.ROOT, p))]
    git("add", "--", *paths)
    nothing_staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=common.ROOT).returncode == 0
    if nothing_staged:
        print("nothing new to commit")
    else:
        git("commit", "-m", args.message or "{}: new runs".format(args.algo))
    git("pull", "--rebase", "--autostash")
    git("push")
    print("\nshared. teammates: git pull, then python -m src.leaderboard")


if __name__ == "__main__":
    main()
