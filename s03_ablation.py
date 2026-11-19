"""
S03 - Systematic ablation on cached frozen-backbone features  (R2.4, R2.6, R2.3 independent runs)

Variants (all trained with the same head, optimiser, augmentation, early stopping and seeds):
  Branch ablation     : M, R, V, M+R, M+V, R+V, M+R+V (full MRV-Net head)
  Fusion-component    : on M+R+V  ->  no_bn | conv1x1 | no_conv | gap_head | no_dropout
  Prediction-level    : soft-voting and hard-voting ensembles of the single-branch models M, R, V
                        (late fusion, i.e. averaging/voting of independently trained classifiers)
Model selection uses ONLY the validation set; the test set is predicted once at the end of each run.

Usage
  python s03_ablation.py                      # everything, all seeds in config.SEEDS
  python s03_ablation.py --only M R V M+R+V   # subset
"""
import os
# Use the PyTorch backend of Keras 3 when PyTorch can see a CUDA GPU (TensorFlow >= 2.11 has no GPU
# support on native Windows). Override with:  set KERAS_BACKEND=tensorflow
if "KERAS_BACKEND" not in os.environ:
    try:
        import torch
        if torch.cuda.is_available():
            os.environ["KERAS_BACKEND"] = "torch"
    except ImportError:
        pass

import os
import time
import argparse

import numpy as np
import pandas as pd

import config as C
from common import to_numpy, ensure_dirs, set_seed, keras, tf, metrics_from_preds, save_json

L = keras.layers
ABBR = {"M": "MobileNetV2", "R": "ResNet50", "V": "VGG16"}
BRANCH_CONFIGS = ["M", "R", "V", "M+R", "M+V", "R+V", "M+R+V"]
COMPONENT_CONFIGS = ["M+R+V|no_bn", "M+R+V|conv1x1", "M+R+V|no_conv", "M+R+V|gap_head", "M+R+V|no_dropout"]


def build_head(in_ch, variant="full", g=8):
    inp = keras.Input((g, g, in_ch))
    x = inp
    if variant != "no_bn":
        x = L.BatchNormalization()(x)
    if variant == "conv1x1":
        x = L.Conv2D(1024, 1, padding="valid", activation=C.FUSION_ACTIVATION)(x)          # 8x8x1024
    elif variant != "no_conv":
        x = L.Conv2D(1024, 2, strides=1, padding=C.FUSION_PADDING, activation=C.FUSION_ACTIVATION)(x)  # 7x7x1024
    if variant == "gap_head":
        x = L.GlobalAveragePooling2D()(x)
    else:
        x = L.MaxPooling2D(2)(x)
        x = L.Flatten()(x)
    rate = 0.0 if variant == "no_dropout" else C.DROPOUT
    x = L.Dropout(rate)(x)
    for u in (1024, 512, 256):
        x = L.Dense(u, activation="relu")(x)
        x = L.Dropout(rate)(x)
    out = L.Dense(3, activation="softmax")(x)
    return keras.Model(inp, out)


def load_feats(split, names):
    return [np.load(os.path.join(C.FEATURE_DIR, f"{split}_{ABBR[n]}.npy"), mmap_mode="r") for n in names]


def make_ds(arrs, y, shuffle, seed):
    n = len(y)
    ch = sum(a.shape[-1] for a in arrs)

    def gen():
        idx = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
        for s in range(0, n, C.BATCH_SIZE):
            b = np.sort(idx[s:s + C.BATCH_SIZE])
            xb = np.concatenate([np.asarray(a[b], dtype=np.float32) for a in arrs], axis=-1)
            yield xb, keras.utils.to_categorical(y[b], 3)

    g = arrs[0].shape[1]
    sig = (tf.TensorSpec((None, g, g, ch), tf.float32), tf.TensorSpec((None, 3), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(2)


def predict(model, arrs):
    n = arrs[0].shape[0]
    out = []
    for s in range(0, n, 64):
        xb = np.concatenate([np.asarray(a[s:s + 64], dtype=np.float32) for a in arrs], axis=-1)
        out.append(to_numpy(model(xb, training=False)))
    return np.concatenate(out)


def evaluate(name, seed, probs, y, test_paths, dedup_paths, extra):
    row = dict(config=name, seed=seed, **extra)
    m = metrics_from_preds(y, probs.argmax(1))
    row.update({f"test_{k}": v for k, v in m.items() if k != "confusion_matrix"})
    row["test_confusion_matrix"] = str(m["confusion_matrix"])
    if dedup_paths is not None:
        keep = np.isin(test_paths, dedup_paths)
        if keep.sum() and keep.sum() < len(keep):
            md = metrics_from_preds(y[keep], probs[keep].argmax(1))
            row["dedup_test_accuracy"] = md["accuracy"]
            row["dedup_test_macro_f1"] = md["macro_f1"]
            row["dedup_n"] = int(keep.sum())
    return row


def save_run_preds(name, seed, probs, y, paths):
    d = os.path.join(C.PRED_DIR, "ablation")
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame({"path": paths, "label": y, "pred": probs.argmax(1)})
    for i in range(3):
        df[f"p{i}"] = probs[:, i]
    df.to_csv(os.path.join(d, f"{name.replace('|', '__').replace('+', '')}_seed{seed}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    args = ap.parse_args()
    ensure_dirs()
    configs = args.only or (BRANCH_CONFIGS + COMPONENT_CONFIGS)
    seeds = args.seeds if args.seeds is not None else C.SEEDS

    mt = np.load(os.path.join(C.FEATURE_DIR, "train_meta.npz"), allow_pickle=True)
    mv = np.load(os.path.join(C.FEATURE_DIR, "val_meta.npz"), allow_pickle=True)
    ms = np.load(os.path.join(C.FEATURE_DIR, "test_meta.npz"), allow_pickle=True)
    ytr, yva, yte = mt["labels"], mv["labels"], ms["labels"]
    test_paths = ms["paths"].astype(str)
    dd = os.path.join(C.RESULTS_DIR, "test_dedup.csv")
    dedup_paths = pd.read_csv(dd).path.values.astype(str) if os.path.exists(dd) else None

    out_csv = os.path.join(C.RESULTS_DIR, "ablation_runs.csv")
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["config"], int(r["seed"])) for r in rows}
    single_probs = {}

    for seed in seeds:
        for cfg in configs:
            branches, variant = (cfg.split("|") + ["full"])[:2]
            names = branches.split("+")
            if (cfg, seed) in done:
                print("skip (done)", cfg, seed)
                if variant == "full" and len(names) == 1:
                    f = os.path.join(C.PRED_DIR, "ablation", f"{names[0]}_seed{seed}.csv")
                    if os.path.exists(f):
                        single_probs[(names[0], seed)] = pd.read_csv(f)[["p0", "p1", "p2"]].values
                continue
            Xtr, Xva, Xte = load_feats("train", names), load_feats("val", names), load_feats("test", names)
            ch = sum(a.shape[-1] for a in Xtr)
            best_run = None
            for lr in getattr(C, "LR_GRID", [1e-3, 1e-4]):   # learning rate selected on validation loss only
                set_seed(seed)
                model = build_head(ch, variant, Xtr[0].shape[1])
                model.compile(optimizer=keras.optimizers.Adam(lr),
                              loss="categorical_crossentropy", metrics=["accuracy"])
                es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE,
                                                   restore_best_weights=True)
                t0 = time.time()
                h = model.fit(make_ds(Xtr, ytr, True, seed), steps_per_epoch=int(np.ceil(len(ytr) / C.BATCH_SIZE)),
                              validation_data=make_ds(Xva, yva, False, 0),
                              validation_steps=int(np.ceil(len(yva) / C.BATCH_SIZE)),
                              epochs=C.MAX_EPOCHS, callbacks=[es], verbose=0)
                vl = np.array(h.history["val_loss"])
                bv = float(np.nanmin(vl)) if np.isfinite(vl).any() else np.inf
                if best_run is None or bv < best_run["val_loss"]:
                    best_run = dict(val_loss=bv, lr=lr, p_te=predict(model, Xte), p_va=predict(model, Xva),
                                    epochs=len(vl), best_epoch=int(np.nanargmin(vl)) + 1 if np.isfinite(vl).any() else 0,
                                    params=int(model.count_params()), secs=time.time() - t0)
                keras.backend.clear_session()
            p_te, p_va, secs = best_run["p_te"], best_run["p_va"], best_run["secs"]
            extra = dict(branches=branches, variant=variant, head_params=best_run["params"],
                         epochs_run=best_run["epochs"], best_epoch=best_run["best_epoch"], selected_lr=best_run["lr"],
                         val_loss=best_run["val_loss"],
                         val_accuracy=float((p_va.argmax(1) == yva).mean()), train_seconds=round(secs, 1))
            rows.append(evaluate(cfg, seed, p_te, yte, test_paths, dedup_paths, extra))
            save_run_preds(cfg, seed, p_te, yte, test_paths)
            if variant == "full" and len(names) == 1:
                single_probs[(names[0], seed)] = p_te
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"{cfg} seed {seed}: test acc {rows[-1]['test_accuracy']:.4f}  ({secs:.0f}s)")
            keras.backend.clear_session()

        # prediction-level (late) fusion of the single-branch classifiers of this seed
        if all((b, seed) in single_probs for b in "MRV"):
            P = np.stack([single_probs[(b, seed)] for b in "MRV"])          # 3 x N x 3
            soft = P.mean(0)
            votes = P.argmax(-1)
            hard = np.zeros_like(soft)
            for i in range(votes.shape[1]):
                c = np.bincount(votes[:, i], minlength=3)
                win = np.flatnonzero(c == c.max())
                k = win[np.argmax(soft[i, win])] if len(win) > 1 else win[0]
                hard[i, k] = 1.0
            for nm, pr in (("Ensemble soft-vote (M,R,V)", soft), ("Ensemble hard-vote (M,R,V)", hard)):
                if (nm, seed) in done:
                    continue
                rows.append(evaluate(nm, seed, pr, yte, test_paths, dedup_paths,
                                     dict(branches="M+R+V", variant="late_fusion", head_params=np.nan)))
                save_run_preds(nm.split(" (")[0].replace(" ", "_"), seed, pr, yte, test_paths)
            pd.DataFrame(rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c.startswith(("test_", "dedup_test_", "val_accuracy", "epochs_run",
                                                  "train_seconds")) and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby("config")[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = df.groupby("config").size()
    agg["head_params"] = df.groupby("config")["head_params"].first()
    order = [c for c in BRANCH_CONFIGS + COMPONENT_CONFIGS + ["Ensemble soft-vote (M,R,V)",
                                                              "Ensemble hard-vote (M,R,V)"] if c in agg.index]
    agg = agg.loc[order]
    agg.to_csv(os.path.join(C.RESULTS_DIR, "ablation_summary.csv"))
    cols = ["test_accuracy_mean", "test_accuracy_std", "test_macro_f1_mean", "test_macro_f1_std", "n_seeds"]
    print(agg[[c for c in cols if c in agg.columns]].round(4))
    save_json(dict(protocol=dict(aug_copies=C.AUG_COPIES, seeds=seeds, max_epochs=C.MAX_EPOCHS,
                                 patience=C.PATIENCE, lr_grid=getattr(C, "LR_GRID", [1e-3, 1e-4]), batch=C.BATCH_SIZE)),
              os.path.join(C.RESULTS_DIR, "s03_protocol.json"))


if __name__ == "__main__":
    main()
