"""
S08b - Paired comparison of MRV-Net with every baseline on the external (PLD) predictions.

Reads results/predictions/external_<set>_<model>.csv written by s08 and reports, for each
baseline: discordant counts, exact McNemar p-value (Holm-adjusted across baselines), the
accuracy difference with a paired bootstrap 95% CI, and the macro-F1 difference.

Output: results/stats_mcnemar_external_<set>.csv

Usage
  python s08b_external_statistics.py               # PLD_all (default)
  python s08b_external_statistics.py --set PLD_test
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

import config as C

REF = "MRV-Net"
BASELINES = ["VGG16", "ResNet50", "MobileNetV2", "Xception", "DenseNet201",
             "EfficientNetB0", "NASNetLarge", "DenseNet121"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="PLD_all")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    def read(model):
        f = os.path.join(C.PRED_DIR, f"external_{a.set}_{model}.csv")
        return pd.read_csv(f)

    ref = read(REF)
    y = ref["label"].to_numpy()
    yr = ref["pred"].to_numpy()
    cr = yr == y
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = rng.integers(0, len(y), size=(a.boot, len(y)))

    rows = []
    for m in BASELINES:
        try:
            d = read(m)
        except FileNotFoundError:
            continue
        assert (d["path"].to_numpy() == ref["path"].to_numpy()).all(), f"image order differs for {m}"
        yo = d["pred"].to_numpy()
        co = yo == y
        b, c = int((cr & ~co).sum()), int((~cr & co).sum())
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        diff = cr.mean() - co.mean()
        bd = cr[idx].mean(1) - co[idx].mean(1)
        f1d = f1_score(y, yr, average="macro") - f1_score(y, yo, average="macro")
        rows.append(dict(comparison=f"{REF} vs {m}", mrv_only=b, other_only=c, p=p, acc_diff=diff,
                         lo=np.percentile(bd, 2.5), hi=np.percentile(bd, 97.5), f1_diff=f1d))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    f = os.path.join(C.RESULTS_DIR, f"stats_mcnemar_external_{a.set}.csv")
    out.to_csv(f, index=False)
    print(out.to_string(index=False))
    print("saved", f)


if __name__ == "__main__":
    main()
