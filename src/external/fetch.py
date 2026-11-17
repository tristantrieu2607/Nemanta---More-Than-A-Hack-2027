"""Download the open datasets used for validation into data/external/ (gitignored).

    python -m src.external.fetch                  # everything that can be fetched automatically
    python -m src.external.fetch --only metrica   # one source

Sources are pinned (git commits / figshare md5) so every teammate benchmarks on the same bytes.
Nothing downloaded here may be committed: data/ is gitignored.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request

from src.common import ROOT

EXTERNAL = os.path.join(ROOT, "data", "external")
USER_AGENT = "mtah2027-validation/1.0"

SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/".format(SKILLCORNER_COMMIT)
SKILLCORNER_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/".format(SKILLCORNER_COMMIT)

METRICA_COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
METRICA_RAW = "https://raw.githubusercontent.com/metrica-sports/sample-data/{}/data/".format(METRICA_COMMIT)
METRICA_FILES = {
    1: ["Sample_Game_1_RawTrackingData_Home_Team.csv", "Sample_Game_1_RawTrackingData_Away_Team.csv",
        "Sample_Game_1_RawEventsData.csv"],
    2: ["Sample_Game_2_RawTrackingData_Home_Team.csv", "Sample_Game_2_RawTrackingData_Away_Team.csv",
        "Sample_Game_2_RawEventsData.csv"],
    3: ["Sample_Game_3_tracking.txt", "Sample_Game_3_metadata.xml", "Sample_Game_3_events.json"],
}

IDSSE_DOI = "10.6084/m9.figshare.28196177"  # Bassek et al. 2025, Sci Data 12:195, CC BY 4.0
IDSSE_URL = "https://springernature.figshare.com/ndownloader/files/{}"
IDSSE_FILES = {  # match -> (matchinformation, positions) figshare file ids, as used by kloppy 3.18
    "J03WPY": (51643487, 51643526), "J03WN1": (51643472, 51643517), "J03WMX": (51643475, 51643514),
    "J03WOH": (51643478, 51643520), "J03WQQ": (51643484, 51643529), "J03WOY": (51643481, 51643523),
    "J03WR9": (51643490, 51643532),
}


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _download(url, path, md5=None):
    """Download url to path unless it is already there (and matches md5). True if downloaded."""
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        if md5 is None or _md5(path) == md5:
            return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = path + ".part"
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
    if os.path.getsize(partial) == 0:
        os.remove(partial)
        raise IOError("empty response from {} (the site blocks scripted downloads)".format(url))
    if md5 is not None and _md5(partial) != md5:
        raise IOError("md5 mismatch for {}".format(url))
    os.replace(partial, path)
    return True


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_metrica():
    for game, names in METRICA_FILES.items():
        for name in names:
            path = os.path.join(EXTERNAL, "metrica", "Sample_Game_{}".format(game), name)
            fresh = _download(METRICA_RAW + "Sample_Game_{}/{}".format(game, name), path)
            print("{} metrica/{}".format("got " if fresh else "have", name))


def fetch_skillcorner():
    target = os.path.join(EXTERNAL, "skillcorner")
    os.makedirs(target, exist_ok=True)
    matches = json.loads(_get(SKILLCORNER_RAW + "matches.json"))
    with open(os.path.join(target, "matches.json"), "w") as handle:
        json.dump(matches, handle, indent=1)
    for match in matches:
        mid = str(match["id"])
        for suffix, base in (("_match.json", SKILLCORNER_RAW),
                             ("_phases_of_play.csv", SKILLCORNER_RAW),
                             ("_tracking_extrapolated.jsonl", SKILLCORNER_LFS)):
            name = mid + suffix
            fresh = _download(base + "matches/{}/{}".format(mid, name), os.path.join(target, mid, name))
            print("{} skillcorner/{}".format("got " if fresh else "have", name))


def fetch_idsse():
    """7 x (match information, positions) XML, ~1.5 GB. Figshare is blocked on some networks
    (e.g. sandboxes); then download by hand, see validation/README.md."""
    target = os.path.join(EXTERNAL, "idsse")
    for match, (meta, positions) in IDSSE_FILES.items():
        for kind, file_id in (("matchinformation", meta), ("positions", positions)):
            name = "{}_{}.xml".format(match, kind)
            try:
                fresh = _download(IDSSE_URL.format(file_id), os.path.join(target, name))
            except Exception as error:  # network policy, proxy, offline
                print("could not download idsse/{} ({}).\nGet the files of https://doi.org/{} by hand"
                      " into {} -- see validation/README.md".format(name, error, IDSSE_DOI,
                                                                   os.path.relpath(target, ROOT)))
                return False
            print("{} idsse/{}".format("got " if fresh else "have", name))
    return True


SOURCES = {"metrica": fetch_metrica, "skillcorner": fetch_skillcorner, "idsse": fetch_idsse}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=sorted(SOURCES))
    args = parser.parse_args()
    for name, fetch in SOURCES.items():
        if args.only in (None, name):
            fetch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
