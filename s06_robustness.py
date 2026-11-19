"""
S06 - Corruption-sensitivity analysis on the independent test set  (R1.5, R2.7)

Nine corruption families x three severities, applied to the 300 test images only
(no retraining): Gaussian noise, Gaussian blur, motion blur, darkening, brightening,
low contrast, JPEG compression, leaf occlusion, cluttered (field-like) background.
Reports accuracy, macro-F1, relative accuracy (corrupted / clean) and mean corruption
accuracy (mCA) for MRV-Net and every available saved baseline.
"""
import os

import numpy as np
import pandas as pd

import config as C
from common import (ensure_dirs, load_split, load_subset, load_model, available_baselines, predict_probs,
                    metrics_from_preds, corrupt, CORRUPTIONS)


def main():
    ensure_dirs()
    split = load_split()
    x, y, _ = load_subset(split, "test")
    keys = [("clean", 0)] + [(k, s) for k in CORRUPTIONS for s in (1, 2, 3)]

    def make(kind, sev):
        # identical corrupted images for every model (seed depends only on corruption and severity)
        if kind == "clean":
            return x
        rng = np.random.default_rng([C.RANDOM_STATE, CORRUPTIONS.index(kind), sev])
        return np.stack([corrupt(im, kind, sev, rng) for im in x])

    # example grid
    try:
        from PIL import Image
        t = 128
        idx = 0
        grid = Image.new("RGB", (t * 4, t * len(CORRUPTIONS)), "white")
        for r, kind in enumerate(CORRUPTIONS):
            for c, sev in enumerate((0, 1, 2, 3)):
                im = x[idx] if sev == 0 else corrupt(x[idx], kind, sev, np.random.default_rng(sev))
                grid.paste(Image.fromarray(im.astype(np.uint8)).resize((t, t)), (c * t, r * t))
        grid.save(os.path.join(C.RESULTS_DIR, "figures", "corruption_examples.png"))
    except Exception as e:
        print("grid failed:", e)

    models = {"MRV-Net": (C.MRVNET_PATH, C.MRVNET_INPUT_MODE, C.IMG_SIZE)}
    for n, p in available_baselines().items():
        models[n] = (p, C.BASELINE_INPUT_MODE[n], C.BASELINE_IMG_SIZE.get(n, C.IMG_SIZE))
    rows = []
    for name, (path, mode, size) in models.items():
        m = load_model(path)
        for kind, sev in keys:
            xv = make(kind, sev)
            if size != C.IMG_SIZE:
                import tensorflow as tf
                xv = tf.image.resize(xv, (size, size)).numpy()
            pr = predict_probs(m, xv, mode)
            met = metrics_from_preds(y, pr.argmax(1))
            rows.append(dict(model=name, corruption=kind, severity=sev, accuracy=met["accuracy"],
                             macro_f1=met["macro_f1"], recall_late_blight=met["recall_Late blight"],
                             recall_healthy=met["recall_Healthy"]))
        print(name, "done")
    df = pd.DataFrame(rows)
    clean = df[df.corruption == "clean"].set_index("model").accuracy
    df["relative_accuracy"] = df.apply(lambda r: r.accuracy / clean[r.model], axis=1)
    df.to_csv(os.path.join(C.RESULTS_DIR, "robustness_long.csv"), index=False)
    cor = df[df.corruption != "clean"]
    summ = cor.groupby("model").agg(mCA=("accuracy", "mean"), mean_macro_f1=("macro_f1", "mean"),
                                    mean_relative_accuracy=("relative_accuracy", "mean"),
                                    worst_accuracy=("accuracy", "min"))
    summ["clean_accuracy"] = clean
    by_kind = cor.pivot_table(index="corruption", columns="model", values="accuracy", aggfunc="mean")
    summ.to_csv(os.path.join(C.RESULTS_DIR, "robustness_summary.csv"))
    by_kind.to_csv(os.path.join(C.RESULTS_DIR, "robustness_by_corruption.csv"))
    print(summ.round(4))
    print(by_kind.round(4))


if __name__ == "__main__":
    main()
