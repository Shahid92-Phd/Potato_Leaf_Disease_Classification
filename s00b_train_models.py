"""
S00b - Train MRV-Net and the eight individual transfer-learning models under ONE protocol.

Every model is trained with the same protocol:

  input 224 x 224 x 3 (256 x 256 images resized, bilinear) | backbone-specific ImageNet preprocessing
  inside each model | frozen ImageNet backbones | categorical cross-entropy | Adam with the learning rate selected from
  {1e-3, 1e-4} by validation loss | batch 32 |
  on-the-fly augmentation of TRAINING images only (rotation 10 deg, shifts/zoom 0.2, shear 0.2 deg,
  horizontal flip) | at most 100 epochs | early stopping on VALIDATION loss (patience 15) with the best
  weights restored | the test set is never used here.

Outputs
  models/<name>.keras (or .h5 for Keras 2)      trained models used by s01-s08
  results/histories/<name>_history.csv           epoch-wise curves (s09 re-plots Figs 7-8)
  results/training_log.csv                       epochs run, best epoch, best val acc/loss, seconds

Usage
  python s00b_train_models.py                    # MRV-Net + all 8 baselines
  python s00b_train_models.py --models MRV-Net   # only MRV-Net
  python s00b_train_models.py --models MRV-Net ResNet50 VGG16 MobileNetV2
Already-trained models are skipped (use --overwrite to retrain).
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
import shutil
import argparse

import numpy as np
import pandas as pd

import config as C
from common import (ensure_dirs, set_seed, keras, tf, load_split, load_images, random_augment,
                    build_mrvnet, build_baseline, BASELINES, save_json, gpu_name)

ALL = ["MRV-Net"] + list(BASELINES)


def batches(x, y, augment, seed):
    rng = np.random.default_rng(seed)
    n = len(y)

    def gen():
        idx = rng.permutation(n) if augment else np.arange(n)
        for s in range(0, n, C.BATCH_SIZE):
            b = idx[s:s + C.BATCH_SIZE]
            xb = np.stack([random_augment(x[i], rng) for i in b]) if augment else x[b]
            yield xb.astype(np.float32), keras.utils.to_categorical(y[b], 3)

    sig = (tf.TensorSpec((None, C.IMG_SIZE, C.IMG_SIZE, 3), tf.float32), tf.TensorSpec((None, 3), tf.float32))
    return tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(2)


def free_gpu():
    """Release memory between models (PyTorch keeps cached CUDA memory otherwise)."""
    import gc
    keras.backend.clear_session()
    gc.collect()
    if keras.backend.backend() == "torch":
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def path_for(name):
    return C.MRVNET_PATH if name == "MRV-Net" else C.BASELINE_PATHS[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ALL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="retrain models that already exist")
    args = ap.parse_args()
    ensure_dirs()
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join(C.RESULTS_DIR, "histories"), exist_ok=True)
    print("Device:", gpu_name(), "| Keras", keras.__version__, "| backend:", keras.backend.backend())
    if gpu_name() == "CPU only":
        print("WARNING: no GPU found. MRV-Net takes roughly 5 min per epoch on a CPU; consider a GPU "
              "(e.g. a Kaggle notebook with GPU enabled).")

    split = load_split()
    xtr, ytr, _ = load_images_split(split, "train")
    xva, yva, _ = load_images_split(split, "val")
    log_f = os.path.join(C.RESULTS_DIR, "training_log.csv")
    log = pd.read_csv(log_f).to_dict("records") if os.path.exists(log_f) else []

    for name in args.models:
        out = path_for(name)
        if os.path.exists(out) and not args.overwrite:
            print(f"{name}: {out} exists, skipping (use --overwrite to retrain)")
            continue
        steps = int(np.ceil(len(ytr) / C.BATCH_SIZE))
        vsteps = int(np.ceil(len(yva) / C.BATCH_SIZE))
        cands = []
        for lr in getattr(C, "LR_GRID", [1e-3, 1e-4]):   # learning rate selected on the VALIDATION loss only
            set_seed(args.seed)
            model = build_mrvnet(C.WEIGHTS) if name == "MRV-Net" else build_baseline(name, C.WEIGHTS)
            model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                          metrics=["accuracy"])
            hist_f = os.path.join(C.RESULTS_DIR, "histories", f"{name}_lr{lr:g}_history.csv")
            cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.PATIENCE,
                                                 restore_best_weights=True),
                   keras.callbacks.CSVLogger(hist_f)]
            print(f"\n=== {name}  learning rate {lr:g} ===")
            t0 = time.time()
            h = model.fit(batches(xtr, ytr, True, args.seed), steps_per_epoch=steps,
                          validation_data=batches(xva, yva, False, 0), validation_steps=vsteps,
                          epochs=C.MAX_EPOCHS, callbacks=cbs, verbose=2)
            secs = time.time() - t0
            vl = np.array(h.history["val_loss"])
            best = int(np.nanargmin(vl)) if np.isfinite(vl).any() else 0
            cand_path = out.replace(C.MODEL_EXT, f"_lr{lr:g}{C.MODEL_EXT}")
            model.save(cand_path)
            cands.append(dict(lr=lr, path=cand_path, hist=hist_f, epochs_run=len(vl), best_epoch=best + 1,
                              best_val_loss=float(vl[best]), best_val_accuracy=float(h.history["val_accuracy"][best]),
                              seconds=secs, params=int(model.count_params())))
            print(f"{name} lr {lr:g}: best epoch {best + 1}/{len(vl)}, val loss {vl[best]:.4f}, "
                  f"val acc {h.history['val_accuracy'][best]:.4f}, {secs / 60:.1f} min")
            del model, h
            free_gpu()
        sel = min(cands, key=lambda c: c["best_val_loss"] if np.isfinite(c["best_val_loss"]) else np.inf)
        shutil.copyfile(sel["path"], out)
        shutil.copyfile(sel["hist"], os.path.join(C.RESULTS_DIR, "histories", f"{name}_history.csv"))
        row = dict(model=name, selected_lr=sel["lr"], epochs_run=sel["epochs_run"], best_epoch=sel["best_epoch"],
                   best_val_loss=sel["best_val_loss"], best_val_accuracy=sel["best_val_accuracy"],
                   train_seconds=round(sel["seconds"], 1), total_params=sel["params"], device=gpu_name(),
                   seed=args.seed)
        for c in cands:
            row[f"val_loss_lr{c['lr']:g}"] = c["best_val_loss"]
            row[f"val_acc_lr{c['lr']:g}"] = c["best_val_accuracy"]
        log = [r for r in log if r["model"] != name] + [row]
        pd.DataFrame(log).to_csv(log_f, index=False)
        print(f">>> {name}: selected learning rate {sel['lr']:g} (validation loss {sel['best_val_loss']:.4f}); "
              f"saved {out}")
    save_json(dict(models=args.models, seed=args.seed, augment=C.AUGMENT, img_size=C.IMG_SIZE,
                   max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE, lr_grid=getattr(C, "LR_GRID", [1e-3, 1e-4]), batch=C.BATCH_SIZE),
              os.path.join(C.RESULTS_DIR, "s00b_protocol.json"))


def load_images_split(split, name):
    d = split[split.split == name].reset_index(drop=True)
    return load_images(d.path.tolist()), d.label.values.astype(int), d


if __name__ == "__main__":
    main()
