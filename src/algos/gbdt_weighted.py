"""gbdt_final trained with extra weight on short possessions and on frames next to a change.

spellF1 counts a 1.5 s possession as much as a 40 s one, but a per-frame loss gives it 1/25 of the
weight. Two public teams weight the loss towards where possessions are won or lost (Barca-Hackathon:
x2 on possessions under 4 s and around changes; False9: x2 near changes). Here: `short_weight` on
training frames inside a possession shorter than 4 s, `edge_weight` on frames within 12 frames
(0.5 s) of a label change. Decode and the kickoff-wait rule are those of gbdt_final.
"""
from src.algos import gbdt_final as GF

PARAMS = dict(GF.PARAMS, short_weight=2.0, edge_weight=2.0)
fit = GF.fit
predict = GF.predict
