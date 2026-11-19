"""
S02 - Cache frozen-backbone feature maps (7 x 7 x C at 224 input) for train / val / test.

Because the three backbones are frozen (ImageNet weights, trainable=False), their outputs are a
fixed function of the input image. Caching them once lets every ablation variant, the
prediction-level ensembles and the repeated-seed runs be trained in minutes (head only).

Training images: the original + AUG_COPIES augmented copies (Table 4 augmentation).
Validation and test images: unaugmented.
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

import numpy as np

import config as C
from common import (to_numpy, ensure_dirs, load_split, load_image, preprocess, random_augment, load_model,
                    nested_backbones, keras, save_json)


def get_backbones():
    bb = {}
    if os.path.exists(C.MRVNET_PATH):
        try:
            bb = nested_backbones(load_model(C.MRVNET_PATH))
        except Exception as e:
            print("Could not read backbones from the saved model:", e)
    ctor = {"MobileNetV2": keras.applications.MobileNetV2, "ResNet50": keras.applications.ResNet50,
            "VGG16": keras.applications.VGG16}
    for name in C.BRANCH_ORDER:
        if name not in bb:
            print(f"{name}: building from keras.applications with ImageNet weights")
            bb[name] = ctor[name](include_top=False, weights="imagenet",
                                  input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3))
        else:
            print(f"{name}: taken from the saved MRV-Net ({bb[name].name})")
        bb[name].trainable = False
    print("Keras backend:", keras.backend.backend())
    return bb


def main():
    ensure_dirs()
    df = load_split()
    bb = get_backbones()
    rng = np.random.default_rng(C.RANDOM_STATE)
    meta = {}
    for split in ("train", "val", "test"):
        d = df[df.split == split].reset_index(drop=True)
        reps = 1 + (C.AUG_COPIES if split == "train" else 0)
        n = len(d) * reps
        labels = np.repeat(d.label.values, reps)
        src = np.repeat(np.arange(len(d)), reps)        # index of the source image
        aug = np.tile(np.arange(reps), len(d))           # 0 = original
        mm = {}
        for name in C.BRANCH_ORDER:
            ch = C.BRANCH_CHANNELS[name]
            fn = os.path.join(C.FEATURE_DIR, f"{split}_{name}.npy")
            mm[name] = np.lib.format.open_memmap(fn, mode="w+", dtype=np.float16, shape=(n, C.IMG_SIZE // 32, C.IMG_SIZE // 32, ch))
        t0 = time.time()
        bs = C.FEATURE_BATCH
        buf, pos = [], 0

        def flush(buf, pos):
            x = np.stack(buf)
            for name in C.BRANCH_ORDER:
                f = bb[name](preprocess(x, C.BRANCH_PREPROCESS[name]), training=False)
                f = to_numpy(f)
                g = C.IMG_SIZE // 32
                assert f.shape[1:3] == (g, g), f"{name} output {f.shape} (expected {g}x{g} for {C.IMG_SIZE} input)"
                mm[name][pos:pos + len(x)] = f.astype(np.float16)
            return pos + len(x)

        for i, p in enumerate(d.path):
            img = load_image(p)
            for r in range(reps):
                buf.append(img if r == 0 else random_augment(img, rng))
                if len(buf) == bs:
                    pos = flush(buf, pos)
                    buf = []
        if buf:
            pos = flush(buf, pos)
        for name in C.BRANCH_ORDER:
            mm[name].flush()
        np.savez(os.path.join(C.FEATURE_DIR, f"{split}_meta.npz"), labels=labels, src=src, aug=aug,
                 paths=d.path.values)
        meta[split] = dict(n_samples=int(n), n_images=int(len(d)), seconds=round(time.time() - t0, 1))
        print(split, meta[split])
    save_json(meta, os.path.join(C.RESULTS_DIR, "s02_cache_meta.json"))


if __name__ == "__main__":
    main()
