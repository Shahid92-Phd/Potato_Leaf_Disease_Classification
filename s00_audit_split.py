"""
S00 - Split export and duplicate / near-duplicate audit  (R2.1, R2.7, dataset provenance)

Outputs
  results/split.csv                      exact file-level split (supplementary material)
  results/split_counts.csv               images per class per subset
  results/image_sizes.csv                original image sizes (resolves the 256x256 statement)
  results/duplicates_exact.csv           byte-identical files (MD5)
  results/duplicates_near.csv            near-duplicates incl. flipped/rotated copies (pHash)
  results/test_dedup.csv                 test images with NO near-duplicate in train/val
  results/figures/near_duplicates.png    contact sheet for visual confirmation
  results/s00_summary.json
"""
import os
import hashlib
import itertools

import numpy as np
import pandas as pd
from PIL import Image

import config as C
from common import ensure_dirs, list_images, save_json

HAMMING_THRESHOLD = 6   # of 64 bits; <=6 is a conservative near-duplicate criterion


ALIASES = {"train": ("train", "training"), "val": ("valid", "val", "validation"), "test": ("test", "testing")}


def find_dir(parent, wanted, aliases=()):
    """Case-insensitive folder lookup (also accepts common aliases)."""
    if not os.path.isdir(parent):
        raise FileNotFoundError(parent)
    names = {n.lower(): n for n in os.listdir(parent) if os.path.isdir(os.path.join(parent, n))}
    for w in (wanted,) + tuple(aliases):
        if w.lower() in names:
            return os.path.join(parent, names[w.lower()])
    raise FileNotFoundError(f"{wanted} not found in {parent}; found {sorted(names.values())}")


def build_split():
    rows = []
    if C.SPLIT_SOURCE == "folders":
        for split, sub in C.SPLIT_FOLDER_NAMES.items():
            sdir = find_dir(C.DATA_ROOT, sub, ALIASES[split])
            for lab, cdir in enumerate(C.CLASS_DIRS):
                folder = find_dir(sdir, cdir)
                rows += [dict(path=p, label=lab, split=split) for p in list_images(folder)]
    elif C.SPLIT_SOURCE == "csv":
        d = pd.read_csv(C.EXISTING_SPLIT_CSV)
        assert {"path", "label", "split"} <= set(d.columns)
        rows = d[["path", "label", "split"]].to_dict("records")
    else:
        raise ValueError("SPLIT_SOURCE must be 'folders' or 'csv'")
    return pd.DataFrame(rows)


def phash(img, hash_size=8, highfreq=4):
    from scipy.fftpack import dct
    g = img.convert("L").resize((hash_size * highfreq, hash_size * highfreq), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float64)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:hash_size, :hash_size]
    return (d > np.median(d)).flatten()


def dihedral(img):
    out = []
    for k in range(4):
        r = img.rotate(90 * k, expand=True)
        out += [r, r.transpose(Image.FLIP_LEFT_RIGHT)]
    return out


def main():
    ensure_dirs()
    df = build_split()
    df.to_csv(C.SPLIT_CSV, index=False)
    counts = df.pivot_table(index="split", columns="label", values="path", aggfunc="count", fill_value=0)
    counts.columns = [C.CLASS_NAMES[i] for i in counts.columns]
    counts["Total"] = counts.sum(1)
    counts.to_csv(os.path.join(C.RESULTS_DIR, "split_counts.csv"))
    print(counts)

    md5, sizes, hashes = [], [], []
    for p in df.path:
        with open(p, "rb") as f:
            md5.append(hashlib.md5(f.read()).hexdigest())
        im = Image.open(p).convert("RGB")
        sizes.append(im.size)
        hashes.append(np.stack([phash(v) for v in dihedral(im)]))  # 8 x 64
    df["md5"] = md5
    df["width"] = [s[0] for s in sizes]
    df["height"] = [s[1] for s in sizes]
    df[["path", "width", "height"]].to_csv(os.path.join(C.RESULTS_DIR, "image_sizes.csv"), index=False)

    exact = df[df.duplicated("md5", keep=False)].sort_values("md5")
    exact.to_csv(os.path.join(C.RESULTS_DIR, "duplicates_exact.csv"), index=False)

    # near duplicates: compare identity hash of i with all 8 dihedral hashes of j
    H0 = np.stack([h[0] for h in hashes]).astype(np.uint8)          # N x 64
    HD = np.stack(hashes).astype(np.uint8)                           # N x 8 x 64
    N = len(df)
    pairs = []
    for i in range(N):
        d = (HD[i + 1:] != H0[i][None, None, :]).sum(-1).min(-1) if i + 1 < N else np.array([])
        for j in np.where(d <= HAMMING_THRESHOLD)[0]:
            jj = i + 1 + j
            pairs.append(dict(path_a=df.path[i], split_a=df.split[i], label_a=df.label[i],
                              path_b=df.path[jj], split_b=df.split[jj], label_b=df.label[jj],
                              hamming=int(d[j])))
    near = pd.DataFrame(pairs, columns=["path_a", "split_a", "label_a", "path_b", "split_b", "label_b", "hamming"])
    near["cross_split"] = near.split_a != near.split_b
    near["label_conflict"] = near.label_a != near.label_b
    near.to_csv(os.path.join(C.RESULTS_DIR, "duplicates_near.csv"), index=False)

    # test images that have a near duplicate in train or val
    leak = set()
    for _, r in near[near.cross_split].iterrows():
        if r.split_a == "test" and r.split_b in ("train", "val"):
            leak.add(r.path_a)
        if r.split_b == "test" and r.split_a in ("train", "val"):
            leak.add(r.path_b)
    test = df[df.split == "test"]
    dedup = test[~test.path.isin(leak)]
    dedup[["path", "label", "split"]].to_csv(os.path.join(C.RESULTS_DIR, "test_dedup.csv"), index=False)

    # contact sheet (up to 24 cross-split pairs, else any pairs)
    show = near[near.cross_split] if near.cross_split.any() else near
    show = show.sort_values("hamming").head(24)
    if len(show):
        t = 96
        sheet = Image.new("RGB", (t * 2 + 10, t * len(show)), "white")
        for k, (_, r) in enumerate(show.iterrows()):
            sheet.paste(Image.open(r.path_a).convert("RGB").resize((t, t)), (0, k * t))
            sheet.paste(Image.open(r.path_b).convert("RGB").resize((t, t)), (t + 10, k * t))
        sheet.save(os.path.join(C.RESULTS_DIR, "figures", "near_duplicates.png"))

    per_class_within = near.groupby("label_a").size().to_dict() if len(near) else {}
    summary = dict(
        n_images=N,
        unique_sizes=sorted(set(map(tuple, sizes))),
        n_exact_duplicate_files=int(len(exact)),
        n_near_duplicate_pairs=int(len(near)),
        n_near_duplicate_pairs_by_class={C.CLASS_NAMES[int(k)]: int(v) for k, v in per_class_within.items()},
        n_cross_split_pairs=int(near.cross_split.sum()) if len(near) else 0,
        n_label_conflicts=int(near.label_conflict.sum()) if len(near) else 0,
        n_test_images=int(len(test)),
        n_test_with_train_or_val_near_duplicate=len(leak),
        n_test_dedup=int(len(dedup)),
        hamming_threshold=HAMMING_THRESHOLD,
    )
    save_json(summary, os.path.join(C.RESULTS_DIR, "s00_summary.json"))
    for k, v in summary.items():
        print(f"{k}: {v}")
    if summary["n_cross_split_pairs"]:
        print("\n!! Cross-split near-duplicates found. Inspect results/figures/near_duplicates.png;"
              " all later scripts also report metrics on results/test_dedup.csv.")


if __name__ == "__main__":
    main()
