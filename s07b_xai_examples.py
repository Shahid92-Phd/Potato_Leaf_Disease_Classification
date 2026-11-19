"""
S07b - Representative SHAP and Grad-CAM panels (Figs. 16 and 17 of the article).

For each class, the correctly classified test image whose fusion-layer Grad-CAM leaf energy is
closest to the class median (results/xai_per_image.csv, written by s07) is shown:
  Fig16_shap_examples.png     image | SHAP for Early blight | Late blight | Healthy outputs
  Fig17_gradcam_examples.png  image | Grad-CAM (fusion layer) | MobileNetV2 | ResNet50 | VGG16 branches
SHAP: Partition explainer on the class logits, blur(32,32) image masker, 500 evaluations per image.
Grad-CAM: logit-based, as in s07_xai_quant.py.

Usage: python s07b_xai_examples.py
"""
import os
import re

import numpy as np
import pandas as pd

import config as C
from common import load_model, load_image, preprocess, keras, tf
from s07_xai_quant import build_grad_models, gradcams, overlay

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_EVALS = 500


def local_path(p):
    """Map a path recorded on another machine to this copy of the repository."""
    if os.path.exists(p):
        return p
    q = p.replace("\\", "/")
    m = re.search(r"/(dataset|data/Potato)/(.*)$", q)
    for base in (os.path.join(HERE, "dataset"), os.path.join(HERE, "data", "Potato"), C.DATA_ROOT):
        cand = os.path.join(base, m.group(2)) if m else q
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(p)


def pick(df):
    out = []
    for c in range(3):
        s = df[(df.label == c) & (df.correct == 1)]
        k = (s.leaf_energy - s.leaf_energy.median()).abs().idxmin()
        out.append(s.loc[k])
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    df = pd.read_csv(os.path.join(C.RESULTS_DIR, "xai_per_image.csv"))
    rows = pick(df)
    imgs = np.stack([load_image(local_path(r.path)) for r in rows]).astype(np.float32)
    m = load_model(C.MRVNET_PATH)
    gm, _ = build_grad_models(m)
    probs = np.asarray(m(preprocess(imgs, C.MRVNET_INPUT_MODE), training=False))
    print("selected:", [os.path.basename(r.path) for r in rows], "pred:", probs.argmax(1))

    # ---------------- Grad-CAM panel
    cols = ["Image", "Grad-CAM (fusion)"] + [f"{b} branch" for b in C.BRANCH_ORDER]
    fig, ax = plt.subplots(3, len(cols), figsize=(2.2 * len(cols), 6.9))
    for i in range(3):
        cls = int(probs[i].argmax())
        cams, _ = gradcams(gm, preprocess(imgs[i:i + 1], C.MRVNET_INPUT_MODE)[0], cls)
        tiles = [imgs[i].astype(np.uint8), overlay(imgs[i], cams["fusion"])] + \
                [overlay(imgs[i], cams[b]) for b in C.BRANCH_ORDER]
        for j, t in enumerate(tiles):
            ax[i, j].imshow(t)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(cols[j], fontsize=9)
        ax[i, 0].set_ylabel(f"{C.CLASS_NAMES[i]}\n(p = {probs[i, cls]:.2f})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS_DIR, "figures", "Fig17_gradcam_examples.png"), dpi=200)
    plt.close(fig)

    # ---------------- SHAP panel (all three class outputs)
    # explain the pre-softmax class scores: probabilities of confidently classified images are saturated
    last = m.layers[-1]
    pen = keras.Model(m.inputs, last.input)
    f = lambda z: np.asarray(tf.matmul(pen(preprocess(np.asarray(z, np.float32), C.MRVNET_INPUT_MODE), training=False),
                                       last.kernel) + last.bias)
    masker = shap.maskers.Image("blur(32,32)", imgs[0].shape)
    expl = shap.Explainer(f, masker, output_names=C.CLASS_NAMES)
    sv = expl(imgs, max_evals=MAX_EVALS, batch_size=C.SHAP_BATCH, silent=True)
    v = np.asarray(sv.values)                      # (n, H, W, 3 channels, 3 outputs)
    vals = v.sum(axis=3)                           # (n, H, W, 3 outputs) in class order 0,1,2
    lim = np.percentile(np.abs(vals), 99.5)
    fig, ax = plt.subplots(3, 4, figsize=(9.2, 7.2))
    for i in range(3):
        ax[i, 0].imshow(imgs[i].astype(np.uint8))
        ax[i, 0].set_ylabel(C.CLASS_NAMES[i], fontsize=9)
        for c in range(3):
            ax[i, c + 1].imshow(imgs[i].astype(np.uint8).mean(-1), cmap="gray", alpha=0.35)
            im = ax[i, c + 1].imshow(vals[i, ..., c], cmap="RdYlGn", vmin=-lim, vmax=lim, alpha=0.85)
            if i == 0:
                ax[i, c + 1].set_title(f"SHAP: {C.CLASS_NAMES[c]} output", fontsize=9)
        if i == 0:
            ax[i, 0].set_title("Image", fontsize=9)
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    cax = fig.add_axes([0.3, 0.03, 0.4, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("SHAP value on the class score (logit); green: evidence for the class, red: evidence against", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(os.path.join(C.RESULTS_DIR, "figures", "Fig16_shap_examples.png"), dpi=200)
    plt.close(fig)
    print("saved Fig16_shap_examples.png and Fig17_gradcam_examples.png")


if __name__ == "__main__":
    main()
