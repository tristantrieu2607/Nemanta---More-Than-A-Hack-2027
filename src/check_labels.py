"""Check our hand-label files, and how well our labelling agrees with the official truth.

    python -m src.check_labels

validation/labels/       stretches we labelled for validation: listed with their size.
validation/calibration/  stretches of the PUBLIC 5 minutes (frames 11185-18684) labelled by us
                         without looking at the answer key, scored against it. A low score means
                         our idea of "control" differs from theirs -- fix that before trusting
                         validation numbers.
"""
import glob
import os

from src import common


def summarise(path, codes):
    labelled = codes != common.X_CODE
    seconds = labelled.sum() / common.FPS
    runs = [(c, n) for c, n in _runs(codes[labelled])]
    poss = sum(1 for c, _ in runs if common.pm.LABELS[c] in "HA")
    print("  {:<32} {:6.0f}s labelled  {:3d} possessions".format(os.path.basename(path), seconds, poss))


def _runs(codes):
    if len(codes) == 0:
        return
    start = 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            yield codes[start], i - start
            start = i


def main():
    n_frames = common.source_frames("tactical")
    print("validation/labels:")
    files = sorted(glob.glob(os.path.join(common.LABELS_DIR, "*.csv")))
    for path in files:
        summarise(path, common.read_label_points(path, n_frames))
    merged, files, digest = common.load_validation(n_frames)
    if merged is None:
        print("  (none yet)")
    else:
        print("  merged OK, no overlaps -- validation set {}".format(digest))

    print("\nvalidation/calibration (vs official public ground truth):")
    public = common.public_gt()
    calib = sorted(glob.glob(os.path.join(common.CALIBRATION_DIR, "*.csv")))
    if not calib:
        print("  (none yet)")
    for path in calib:
        ours = common.read_label_points(path, n_frames)
        truth = public.copy()
        truth[ours == common.X_CODE] = common.X_CODE  # score only the stretch we labelled
        s = common.score(ours, truth)
        print("  {:<32} spellF1 {:.3f}  macroF1 {:.3f}  poss ours {} / theirs {}".format(
            os.path.basename(path), s["spell_f1_mean"], s["macro_f1"],
            s["n_pred_spells"], s["n_gt_spells"]))


if __name__ == "__main__":
    main()
