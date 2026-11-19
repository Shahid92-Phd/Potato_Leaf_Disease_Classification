"""
S09 - Re-plot Figs 7 and 8 with correct legends (training vs VALIDATION)  (R2.1)

Input: the saved Keras training history of MRV-Net, any of
  history.csv  (keras.callbacks.CSVLogger; columns epoch, accuracy, loss, val_accuracy, val_loss)
  history.json / history.pkl  (dict of lists from model.fit(...).history)
Usage: python s09_learning_curves.py   (defaults to results/histories/MRV-Net_history.csv from s00b)
"""
import os
import sys
import json
import pickle

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402


def load(p):
    if p.endswith(".csv"):
        return pd.read_csv(p)
    if p.endswith(".json"):
        return pd.DataFrame(json.load(open(p)))
    return pd.DataFrame(pickle.load(open(p, "rb")))


def main():
    h = load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(C.RESULTS_DIR, "histories", "MRV-Net_history.csv"))
    acc = "accuracy" if "accuracy" in h else "acc"
    ep = range(1, len(h) + 1)
    best = int(h["val_loss"].idxmin()) + 1
    os.makedirs(os.path.join(C.RESULTS_DIR, "figures"), exist_ok=True)
    for key, ylab, fn in ((acc, "Accuracy", "Fig7_accuracy.png"), ("loss", "Categorical cross-entropy loss", "Fig8_loss.png")):
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(ep, h[key], label="Training")
        plt.plot(ep, h["val_" + key], label="Validation")
        plt.axvline(best, ls="--", c="grey", lw=0.8, label=f"Restored epoch ({best})")
        plt.xlabel("Epoch"); plt.ylabel(ylab); plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(os.path.join(C.RESULTS_DIR, "figures", fn))
        plt.close()
    print("Saved Fig7_accuracy.png and Fig8_loss.png; best validation-loss epoch:", best)


if __name__ == "__main__":
    main()
