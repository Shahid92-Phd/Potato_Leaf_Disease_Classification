"""
S07 - Quantitative explainability assessment on ALL test images  (R2.5)

Grad-CAM (implementation): target layer = the 2x2 fusion Conv2D (the only layer that receives all
three backbones; 7x7x1024), class = predicted class; channel weights = spatially averaged gradients
of the pre-softmax class score (logit); ReLU; bilinear up-sampling to 224x224; min-max normalisation.
Branch-wise Grad-CAM: the same procedure applied to the channel blocks of the Concatenate output
(MobileNetV2 / ResNet50 / VGG16), giving the spatial evidence carried by each backbone.

Per-image measures (all 300 test images, by class and correct vs misclassified):
  leaf_energy       fraction of saliency inside the automatically segmented leaf (1 - background reliance)
  leaf_area         fraction of the image covered by the leaf (chance level for leaf_energy)
  lesion_energy     fraction of in-leaf saliency on symptomatic tissue (proxy or expert masks)
  pointing_hit      saliency maximum falls on symptomatic tissue (15-px tolerance)
  iou_top20         IoU of the top-20% salient pixels with the symptomatic-tissue mask
  deletion_auc / insertion_auc and the same for a random pixel order (faithfulness)
  stability_*       Spearman correlation of Grad-CAM under small noise, +5% brightness, 4-px shift
SHAP (optional, --shap): Partition explainer with a blur(32,32) image masker on the predicted
class; leaf_energy of positive attributions and Spearman agreement with Grad-CAM.

Usage: python s07_xai_quant.py [--shap] [--limit N]
       Large runs can be split:  --range A B  (writes results/xai_parts/part_A_B.csv), then --merge.
       --reuse-random / --reuse-shap take the random-order AUCs / SHAP columns from an existing
       results/xai_per_image.csv (they do not depend on the Grad-CAM definition).
"""
import os
import argparse

import numpy as np
import pandas as pd
import cv2
from scipy import stats

import config as C
from common import (ensure_dirs, load_split, load_subset, load_model, preprocess, tf, keras,
                    leaf_mask, proxy_lesion_mask)

STEPS = 20
_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz


def build_grad_models(m):
    """Model returning the fusion-conv activations, the concatenated backbone features and the
    pre-softmax class scores (logits). Grad-CAM is computed from the logit of the class, as in
    Selvaraju et al. (2017): gradients of softmax probabilities vanish for saturated predictions."""
    convs = [l for l in m.layers if isinstance(l, keras.layers.Conv2D)]
    concat = [l for l in m.layers if isinstance(l, keras.layers.Concatenate)]
    if not convs or not concat:
        raise RuntimeError("Could not find the top-level fusion Conv2D / Concatenate layers.")
    last = m.layers[-1]
    gm = keras.Model(m.inputs, [convs[-1].output, concat[0].output, last.input])
    gm._logit_layer = last
    return gm, convs[-1].name


def gradcams(gm, x_pre, cls):
    """returns dict of CAMs (fusion and one per branch) and the class probability."""
    last = gm._logit_layer
    xt = tf.convert_to_tensor(x_pre[None])
    with tf.GradientTape() as tape:
        conv, cat, pen = gm(xt, training=False)
        logits = tf.matmul(pen, last.kernel) + last.bias
        score = logits[:, cls]
    gconv, gcat = tape.gradient(score, [conv, cat])

    def cam(A, G):
        w = tf.reduce_mean(G, axis=(1, 2), keepdims=True)
        c = tf.nn.relu(tf.reduce_sum(w * A, -1))[0].numpy()
        c = cv2.resize(c, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        return (c - c.min()) / (c.max() - c.min() + 1e-12)

    out_cams = {"fusion": cam(conv, gconv)}
    s = 0
    for name in C.BRANCH_ORDER:
        ch = C.BRANCH_CHANNELS[name]
        out_cams[name] = cam(cat[..., s:s + ch], gcat[..., s:s + ch])
        s += ch
    return out_cams, float(tf.nn.softmax(logits)[0, cls])


def energy(sal, mask):
    t = sal.sum()
    return float((sal * mask).sum() / t) if t > 0 else np.nan


def del_ins(model, img, sal, cls, rng, orders=("sal", "rand")):
    blur = cv2.GaussianBlur(img, (0, 0), 10)
    n = sal.size
    res = {}
    for tag in orders:
        order = np.argsort(-sal.ravel()) if tag == "sal" else rng.permutation(n)
        batch_d, batch_i = [], []
        for k in range(STEPS + 1):
            m = np.zeros(n, bool)
            m[order[: int(n * k / STEPS)]] = True
            m = m.reshape(sal.shape)[..., None]
            batch_d.append(np.where(m, blur, img))
            batch_i.append(np.where(m, img, blur))
        xb = np.stack(batch_d + batch_i).astype(np.float32)
        p = np.concatenate([np.asarray(model(preprocess(xb[j:j + 8], C.MRVNET_INPUT_MODE), training=False))[:, cls]
                            for j in range(0, len(xb), 8)])
        xs = np.linspace(0, 1, STEPS + 1)
        res[f"deletion_auc_{tag}"] = float(_trapz(p[: STEPS + 1], xs))
        res[f"insertion_auc_{tag}"] = float(_trapz(p[STEPS + 1:], xs))
    return res


def spear(a, b):
    a = cv2.resize(a, (64, 64)).ravel()
    b = cv2.resize(b, (64, 64)).ravel()
    return float(stats.spearmanr(a, b).correlation)


def lesion_metrics(sal, lesion, leaf, prefix):
    out = {}
    if lesion is None or lesion.sum() < 0.005 * max(leaf.sum(), 1):
        return out
    out[f"{prefix}lesion_area_in_leaf"] = float(lesion.sum() / max(leaf.sum(), 1))
    out[f"{prefix}lesion_energy"] = float((sal * lesion).sum() / ((sal * leaf).sum() + 1e-12))
    dil = cv2.dilate(lesion.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    yx = np.unravel_index(np.argmax(sal), sal.shape)
    out[f"{prefix}pointing_hit"] = float(dil[yx])
    thr = np.quantile(sal, 1 - C.TOP_FRACTION)
    top = sal >= thr
    out[f"{prefix}iou_top20"] = float((top & lesion).sum() / ((top | lesion).sum() + 1e-12))
    return out


def run_shap(model, x, idx, cls_pred):
    import shap
    f = lambda z: np.asarray(model(preprocess(np.asarray(z, np.float32), C.MRVNET_INPUT_MODE), training=False))
    masker = shap.maskers.Image("blur(32,32)", x[0].shape)
    expl = shap.Explainer(f, masker, output_names=C.CLASS_NAMES)
    maps = {}
    for i in idx:
        sv = expl(x[i:i + 1], max_evals=C.SHAP_MAX_EVALS, batch_size=C.SHAP_BATCH, silent=True)
        v = sv.values[0][..., int(cls_pred[i])].sum(-1)            # H x W, predicted class
        maps[i] = v
    return maps


def overlay(img, cam):
    hm = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (0.55 * img + 0.45 * hm).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--range", nargs=2, type=int, default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--reuse-random", action="store_true")
    ap.add_argument("--reuse-shap", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    split = load_split()
    x, y, d = load_subset(split, "test")
    if args.limit:
        x, y, d = x[: args.limit], y[: args.limit], d.iloc[: args.limit]
    prev_f = os.path.join(C.RESULTS_DIR, "xai_per_image.csv")
    prev = pd.read_csv(prev_f) if os.path.exists(prev_f) else None
    parts_dir = os.path.join(C.RESULTS_DIR, "xai_parts")
    m = load_model(C.MRVNET_PATH)
    gm, conv_name = build_grad_models(m)
    print("Grad-CAM target layer:", conv_name)
    rng = np.random.default_rng(C.RANDOM_STATE)
    rows, cams_keep = [], {}
    idx_run = range(*args.range) if args.range else range(len(x))
    if args.merge:
        idx_run = []
        rows = pd.concat([pd.read_csv(os.path.join(parts_dir, f)) for f in sorted(os.listdir(parts_dir))]) \
            .sort_values("index").drop(columns="index").to_dict("records")
        assert len(rows) == len(x), f"{len(rows)} merged rows for {len(x)} images"
        for i in range(len(x)):
            if rows[i]["correct"] == 0 or i < 6:
                xp = preprocess(x[i][None], C.MRVNET_INPUT_MODE)[0]
                cams_keep[i] = (x[i], gradcams(gm, xp, rows[i]["pred"])[0], leaf_mask(x[i]))
    for i in idx_run:
        img = x[i]
        xp = preprocess(img[None], C.MRVNET_INPUT_MODE)[0]
        probs = np.asarray(m(xp[None], training=False))[0]
        cls = int(probs.argmax())
        cams, _ = gradcams(gm, xp, cls)
        sal = cams["fusion"]
        leaf = leaf_mask(img)
        lesion = proxy_lesion_mask(img, leaf) if y[i] != 2 else None
        r = dict(path=d.path.iloc[i], label=int(y[i]), pred=cls, correct=int(cls == y[i]),
                 confidence=float(probs[cls]), leaf_area=float(leaf.mean()),
                 leaf_energy=energy(sal, leaf))
        for b in C.BRANCH_ORDER:
            r[f"leaf_energy_{b}"] = energy(cams[b], leaf)
        r.update(lesion_metrics(sal, lesion, leaf, "proxy_"))
        if C.EXPERT_MASK_DIR:
            fn = os.path.join(C.EXPERT_MASK_DIR, os.path.splitext(os.path.basename(r["path"]))[0] + ".png")
            if os.path.exists(fn):
                em = cv2.resize(cv2.imread(fn, 0), (C.IMG_SIZE, C.IMG_SIZE)) > 127
                r.update(lesion_metrics(sal, em, leaf, "expert_"))
        reuse = args.reuse_random and prev is not None and "deletion_auc_rand" in prev
        r.update(del_ins(m, img, sal, cls, rng, orders=("sal",) if reuse else ("sal", "rand")))
        if reuse:
            r["deletion_auc_rand"] = float(prev.deletion_auc_rand.iloc[i])
            r["insertion_auc_rand"] = float(prev.insertion_auc_rand.iloc[i])
        # stability
        noisy = np.clip(img + rng.normal(0, 2.55, img.shape), 0, 255).astype(np.float32)
        bright = np.clip(img * 1.05, 0, 255).astype(np.float32)
        shift = np.roll(img, (4, 4), axis=(0, 1))
        c_n, _ = gradcams(gm, preprocess(noisy[None], C.MRVNET_INPUT_MODE)[0], cls)
        c_b, _ = gradcams(gm, preprocess(bright[None], C.MRVNET_INPUT_MODE)[0], cls)
        c_s, _ = gradcams(gm, preprocess(shift[None], C.MRVNET_INPUT_MODE)[0], cls)
        r["stability_noise"] = spear(sal, c_n["fusion"])
        r["stability_brightness"] = spear(sal, c_b["fusion"])
        r["stability_shift"] = spear(sal[:-4, :-4], np.roll(c_s["fusion"], (-4, -4), axis=(0, 1))[:-4, :-4])
        rows.append(r)
        if cls != y[i] or i < 6:
            cams_keep[i] = (img, cams, leaf)
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(x)}")
    if args.range:
        os.makedirs(parts_dir, exist_ok=True)
        part = pd.DataFrame(rows)
        part.insert(0, "index", list(idx_run))
        part.to_csv(os.path.join(parts_dir, f"part_{args.range[0]:03d}_{args.range[1]:03d}.csv"), index=False)
        print("saved part", args.range)
        return
    df = pd.DataFrame(rows)
    if args.reuse_shap and prev is not None and "shap_leaf_energy" in prev:
        df["shap_leaf_energy"] = prev.shap_leaf_energy.values[: len(df)]

    if args.shap:
        idx = list(range(len(x))) if C.SHAP_ON_ALL_TEST else (
            list(df.index[df.correct == 0]) + list(rng.choice(df.index[df.correct == 1], 30, replace=False)))
        smaps = run_shap(m, x, idx, df.pred.values)
        for i, v in smaps.items():
            pos = np.clip(v, 0, None)
            leaf = leaf_mask(x[i])
            df.loc[i, "shap_leaf_energy"] = energy(pos, leaf)
            g, _ = gradcams(gm, preprocess(x[i][None], C.MRVNET_INPUT_MODE)[0], int(df.pred[i]))  # logit-based
            df.loc[i, "shap_gradcam_spearman"] = spear(pos / (pos.max() + 1e-12), g["fusion"])

    df.to_csv(os.path.join(C.RESULTS_DIR, "xai_per_image.csv"), index=False)
    df["class"] = [C.CLASS_NAMES[k] for k in df.label]
    df["group"] = np.where(df.correct == 1, "correct", "misclassified")
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("label", "pred", "correct")]
    parts = [df[num].mean().rename("All").to_frame().T,
             df.groupby("class")[num].mean(), df.groupby("group")[num].mean()]
    summ = pd.concat(parts)
    summ.insert(0, "n", [len(df)] + list(df.groupby("class").size()) + list(df.groupby("group").size()))
    summ.to_csv(os.path.join(C.RESULTS_DIR, "xai_summary.csv"))
    w = stats.wilcoxon(df.deletion_auc_sal, df.deletion_auc_rand, alternative="less")
    wi = stats.wilcoxon(df.insertion_auc_sal, df.insertion_auc_rand, alternative="greater")
    wl = stats.wilcoxon(df.leaf_energy - df.leaf_area, alternative="greater")
    tests = dict(deletion_sal_lt_random_p=w.pvalue, insertion_sal_gt_random_p=wi.pvalue,
                 leaf_energy_gt_leaf_area_p=wl.pvalue)
    pd.Series(tests).to_csv(os.path.join(C.RESULTS_DIR, "xai_tests.csv"))
    cols = ["n", "leaf_area", "leaf_energy", "proxy_lesion_energy", "proxy_pointing_hit", "proxy_iou_top20",
            "deletion_auc_sal", "deletion_auc_rand", "insertion_auc_sal", "insertion_auc_rand",
            "stability_noise", "stability_brightness", "stability_shift"]
    print(summ[[c for c in cols if c in summ.columns]].round(3).to_string())
    print(tests)

    # gallery: misclassified first, then examples
    keys = sorted(cams_keep, key=lambda k: (rows[k]["correct"], k))[:14]
    t = 160
    from PIL import Image, ImageDraw
    cols_n = 2 + len(C.BRANCH_ORDER)
    g = Image.new("RGB", (t * cols_n, (t + 18) * len(keys)), "white")
    dr = ImageDraw.Draw(g)
    for r_i, k in enumerate(keys):
        img, cams, leaf = cams_keep[k]
        tiles = [img.astype(np.uint8), overlay(img, cams["fusion"])] + [overlay(img, cams[b]) for b in C.BRANCH_ORDER]
        for c_i, tile in enumerate(tiles):
            g.paste(Image.fromarray(tile).resize((t, t)), (c_i * t, r_i * (t + 18) + 18))
        rr = rows[k]
        dr.text((4, r_i * (t + 18) + 2), f"true={C.CLASS_NAMES[rr['label']]}  pred={C.CLASS_NAMES[rr['pred']]}"
                f"  p={rr['confidence']:.2f}   [image | fusion | " + " | ".join(C.BRANCH_ORDER) + "]", fill="black")
    g.save(os.path.join(C.RESULTS_DIR, "figures", "xai_gallery_misclassified_and_examples.png"))


if __name__ == "__main__":
    main()
