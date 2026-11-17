"""External validation: open matches that are not BAR-ATH, made to look like it.

The 3 Nov organiser clarification: no hand labels on the provided game, and every tuned parameter
must be tuned on external data. This package is where that happens.

    fetch.py        download the open datasets into data/external/ (gitignored)
    idsse.py        DFL / IDSSE Bundesliga (Bassek et al. 2025, CC BY 4.0)
    skillcorner.py  SkillCorner Open Data, A-League 2024/25 (MIT)
    metrica.py      Metrica Sports sample data, 3 anonymised games (no licence file)
    labels.py       turn provider possession flags / events into the challenge's possession rule
    degrade.py      make clean tracking look like the BAR-ATH automatic tracking (ATD)
    bench.py        the benchmark: load, degrade, run an algorithm, score with the organisers' scorer
"""

from src import common as _common  # noqa: F401,E402  puts evaluation/ (the organisers' scorer) on sys.path
