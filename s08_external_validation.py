"""
S08 - External validation on independently acquired, field-captured images  (R1.5, R2.7)

The saved models are applied WITHOUT any retraining, fine-tuning or threshold adjustment to
each dataset listed in config.EXTERNAL_SETS (e.g. the PLD dataset from Central Punjab, Pakistan).
Reports accuracy (Wilson 95% CI), macro precision/recall/F1, per-class recall and confusion matrix.
"""
import os

import numpy as np
import pandas as pd

import config as C
from common import (ensure_dirs, list_images, load_images, load_model, available_baselines, predict_probs,
                    metrics_from_preds, wilson_ci, save_json)


def main():
    ensure_dirs()
    models = {"MRV-Net": (C.MRVNET_PATH, C.MRVNET_INPUT_MODE, C.IMG_SIZE)}
    for n, p in available_baselines().items():
        models[n] = (p, C.BASELINE_INPUT_MODE[n], C.BASELINE_IMG_SIZE.get(n, C.IMG_SIZE))
    rows, cms = [], {}
    for ds, spec in C.EXTERNAL_SETS.items():
        paths, labels = [], []
        from s00_audit_split import find_dir
        roots = spec["root"] if isinstance(spec["root"], (list, tuple)) else [spec["root"]]
        ok = True
        for root in roots:
            for lab, cd in enumerate(spec["class_dirs"]):
                try:
                    f = find_dir(root, cd)   # case-insensitive
                except FileNotFoundError as e:
                    print(f"[{ds}] {e}; skipping dataset")
                    ok = False
                    break
                ps = list_images(f)
                paths += ps
                labels += [lab] * len(ps)
            if not ok:
                break
        if not ok:
            paths = []
        if not paths:
            continue
        y = np.array(labels)
        counts = np.bincount(y, minlength=3).tolist()
        print(f"[{ds}] {len(y)} images, per class {counts}")
        cache = {}
        for name, (path, mode, size) in models.items():
            if size not in cache:
                cache[size] = load_images(paths, size)
            m = load_model(path)
            pr = predict_probs(m, cache[size], mode)
            met = metrics_from_preds(y, pr.argmax(1))
            k = int((pr.argmax(1) == y).sum())
            lo, hi = wilson_ci(k, len(y))
            rows.append(dict(dataset=ds, model=name, n=len(y), class_counts=str(counts), accuracy=met["accuracy"],
                             acc_ci_lo=lo, acc_ci_hi=hi, macro_precision=met["macro_precision"],
                             macro_recall=met["macro_recall"], macro_f1=met["macro_f1"],
                             **{f"recall_{c}": met[f"recall_{c}"] for c in C.CLASS_NAMES}))
            cms[f"{ds}|{name}"] = met["confusion_matrix"]
            pd.DataFrame({"path": paths, "label": y, "pred": pr.argmax(1), **{f"p{i}": pr[:, i] for i in range(3)}}
                         ).to_csv(os.path.join(C.PRED_DIR, f"external_{ds}_{name}.csv"), index=False)
            print(f"  {name}: acc {met['accuracy']:.4f} [{lo:.3f}, {hi:.3f}]  macro-F1 {met['macro_f1']:.4f}")
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS_DIR, "external_validation.csv"), index=False)
        save_json(cms, os.path.join(C.RESULTS_DIR, "external_confusion_matrices.json"))


if __name__ == "__main__":
    main()
