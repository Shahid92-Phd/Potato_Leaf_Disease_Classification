"""Shared utilities for the MRV-Net experiments."""
import os
import json
import random
import importlib

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

import config as C  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    for d in (C.RESULTS_DIR, C.PRED_DIR, C.FEATURE_DIR, os.path.join(C.RESULTS_DIR, "figures")):
        os.makedirs(d, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except Exception:
        pass
    if C.DETERMINISTIC_OPS:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def gpu_name():
    if keras.backend.backend() == "torch":
        import torch
        return torch.cuda.get_device_name(0) + " (PyTorch backend)" if torch.cuda.is_available() else "CPU only"
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU only"
    try:
        return tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU")
    except Exception:
        return "GPU"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(IMG_EXT))


def load_split():
    if not os.path.exists(C.SPLIT_CSV):
        raise FileNotFoundError(f"{C.SPLIT_CSV} not found. Run s00_audit_split.py first.")
    return pd.read_csv(C.SPLIT_CSV)


def load_image(path, size=None):
    size = size or C.IMG_SIZE
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def load_images(paths, size=None):
    return np.stack([load_image(p, size) for p in paths]).astype(np.float32)


def load_subset(df, split, size=None):
    d = df[df.split == split].reset_index(drop=True)
    return load_images(d.path.tolist(), size), d.label.values.astype(int), d


# ---------------------------------------------------------------------------
# Preprocessing / model I/O
# ---------------------------------------------------------------------------
_PRE = {
    "vgg16": "vgg16", "vgg19": "vgg19", "resnet50": "resnet50", "resnet": "resnet",
    "mobilenet_v2": "mobilenet_v2", "xception": "xception", "densenet": "densenet",
    "efficientnet": "efficientnet", "nasnet": "nasnet", "inception_v3": "inception_v3",
}


def _app(mode):
    return getattr(keras.applications, _PRE[mode])


def to_numpy(t):
    """Backend-independent conversion (TensorFlow or PyTorch/CUDA tensors)."""
    try:
        return keras.ops.convert_to_numpy(t)
    except Exception:
        return np.asarray(t)


def preprocess(x, mode):
    """x: float32 RGB in [0,255]."""
    if mode == "raw":
        return x
    if mode == "unit":
        return x / 255.0
    return _app(mode).preprocess_input(np.array(x, copy=True))


_reg = getattr(getattr(keras, "saving", None), "register_keras_serializable", None) or keras.utils.register_keras_serializable


@_reg(package="mrvnet")
class Preprocess(keras.layers.Layer):
    """Applies the Keras ImageNet preprocessing of one backbone inside the model (input: RGB 0-255)."""

    def __init__(self, mode="raw", **kw):
        super().__init__(**kw)
        self.mode = mode

    def call(self, x):
        if self.mode == "raw":
            return x
        return _app(self.mode).preprocess_input(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(mode=self.mode)
        return cfg


BASELINES = {  # name: (keras.applications constructor name, preprocessing mode)
    "VGG16": ("VGG16", "vgg16"), "ResNet50": ("ResNet50", "resnet50"), "MobileNetV2": ("MobileNetV2", "mobilenet_v2"),
    "Xception": ("Xception", "xception"), "DenseNet201": ("DenseNet201", "densenet"),
    "EfficientNetB0": ("EfficientNetB0", "efficientnet"), "NASNetLarge": ("NASNetLarge", "nasnet"),
    "DenseNet121": ("DenseNet121", "densenet"),
}


def build_baseline(name, weights=None):
    """Individually trained transfer-learning model, head:
    frozen backbone -> BN -> 2x2 max-pool -> dropout -> flatten -> 1024/512/256 dense (ReLU, dropout) -> softmax."""
    L = keras.layers
    ctor, mode = BASELINES[name]
    s = C.IMG_SIZE
    b = getattr(keras.applications, ctor)(include_top=False, weights=weights, input_shape=(s, s, 3))
    b.trainable = False
    inp = keras.Input((s, s, 3))
    x = b(Preprocess(mode, name=f"pre_{mode}")(inp), training=False)
    x = L.BatchNormalization()(x)
    x = L.MaxPooling2D(2)(x)
    x = L.Dropout(C.DROPOUT)(x)
    x = L.Flatten()(x)
    for u in (1024, 512, 256):
        x = L.Dense(u, activation="relu")(x)
        x = L.Dropout(C.DROPOUT)(x)
    return keras.Model(inp, L.Dense(3, activation="softmax")(x), name=name)


def build_mrvnet(weights=None):
    """MRV-Net as implemented (frozen backbones, per-branch preprocessing inside the model;
    224x224 input -> 7x7 backbone maps -> 2x2 'same' fusion conv -> 7x7x1024 -> 2x2 max-pool -> 3x3x1024 -> 9,216)."""
    L = keras.layers
    app = keras.applications
    ctor = {"MobileNetV2": (app.MobileNetV2, app.mobilenet_v2.preprocess_input),
            "ResNet50": (app.ResNet50, app.resnet50.preprocess_input),
            "VGG16": (app.VGG16, app.vgg16.preprocess_input)}
    inp = keras.Input((C.IMG_SIZE, C.IMG_SIZE, 3))
    feats = []
    for name in C.BRANCH_ORDER:
        f, pre = ctor[name]
        b = f(include_top=False, weights=weights, input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3))
        b.trainable = False
        feats.append(b(Preprocess(C.BRANCH_PREPROCESS[name], name=f"pre_{name}")(inp), training=False))
    x = L.Concatenate()(feats)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1024, 2, strides=1, padding=C.FUSION_PADDING, activation=C.FUSION_ACTIVATION)(x)
    x = L.MaxPooling2D(2)(x)
    x = L.Flatten()(x)
    x = L.Dropout(C.DROPOUT)(x)
    for u in (1024, 512, 256):
        x = L.Dense(u, activation="relu")(x)
        x = L.Dropout(C.DROPOUT)(x)
    return keras.Model(inp, L.Dense(3, activation="softmax")(x), name="MRV_Net")


def load_model(path):
    kw = dict(compile=False, custom_objects={"Preprocess": Preprocess})
    try:
        try:
            return keras.models.load_model(path, safe_mode=False, **kw)
        except TypeError:  # older tf.keras has no safe_mode argument
            return keras.models.load_model(path, **kw)
    except Exception as e:
        if os.path.abspath(path) != os.path.abspath(C.MRVNET_PATH):
            raise
        # Typical cause: Lambda(preprocess_input) layers cannot be deserialised in another
        # Keras/Python version. Rebuild the architecture in code and load the trained weights.
        print(f"Full-model loading failed ({type(e).__name__}); rebuilding MRV-Net and loading weights.")
        m = build_mrvnet()
        m.load_weights(path)
        return m


def available_baselines():
    return {k: v for k, v in C.BASELINE_PATHS.items() if v and os.path.exists(v)}


def predict_probs(model, x_raw, mode, batch=32):
    out = []
    for i in range(0, len(x_raw), batch):
        xb = preprocess(x_raw[i:i + batch], mode)
        out.append(to_numpy(model(xb, training=False)))
    return np.concatenate(out, 0)


def find_top_level(model, cls):
    return [l for l in model.layers if isinstance(l, cls)]


def nested_backbones(model):
    """Return {branch_name: sub-model} by matching output channels."""
    subs = [l for l in model.layers if isinstance(l, keras.Model)]
    found = {}
    for s in subs:
        ch = s.output_shape[-1] if hasattr(s, "output_shape") else s.output.shape[-1]
        for name, c in C.BRANCH_CHANNELS.items():
            if ch == c and name not in found:
                found[name] = s
    return found


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_from_preds(y, yhat, k=3):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    p, r, f, _ = precision_recall_fscore_support(y, yhat, labels=list(range(k)), zero_division=0)
    out = {
        "accuracy": accuracy_score(y, yhat),
        "macro_precision": p.mean(), "macro_recall": r.mean(), "macro_f1": f.mean(),
        "sd_precision": p.std(ddof=1), "sd_recall": r.std(ddof=1), "sd_f1": f.std(ddof=1),  # SD over the 3 classes, n-1 (as in the manuscript)
    }
    for i, n in enumerate(C.CLASS_NAMES):
        out[f"precision_{n}"], out[f"recall_{n}"], out[f"f1_{n}"] = p[i], r[i], f[i]
    out["confusion_matrix"] = confusion_matrix(y, yhat, labels=list(range(k))).tolist()
    return out


def wilson_ci(k, n, z=1.959964):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def save_preds(name, df_split, probs, subdir=""):
    d = os.path.join(C.PRED_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    out = df_split[["path", "label"]].copy()
    for i in range(probs.shape[1]):
        out[f"p{i}"] = probs[:, i]
    out["pred"] = probs.argmax(1)
    fn = os.path.join(d, f"{name}.csv")
    out.to_csv(fn, index=False)
    return fn


# ---------------------------------------------------------------------------
# Augmentation (Table 4): rotation 24 deg, shear 0.2, zoom 0.2, shifts 0.2, flips
# Implemented with OpenCV so it is independent of the Keras version.
# ---------------------------------------------------------------------------
def random_augment(img, rng):
    import cv2
    a = C.AUGMENT
    h, w = img.shape[:2]
    theta = np.deg2rad(rng.uniform(-a["rotation"], a["rotation"]))
    # Keras ImageDataGenerator interprets shear_range in DEGREES (0.2 -> +/-0.2 deg)
    shear = np.deg2rad(rng.uniform(-a["shear"], a["shear"]))
    zx, zy = rng.uniform(1 - a["zoom"], 1 + a["zoom"], 2)
    tx, ty = rng.uniform(-a["shift"], a["shift"]) * w, rng.uniform(-a["shift"], a["shift"]) * h
    cx, cy = w / 2, h / 2
    T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    S = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    Z = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
    T2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]])
    M = (T2 @ R @ S @ Z @ T1)[:2]
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if a["hflip"] and rng.random() < 0.5:
        out = out[:, ::-1]
    if a["vflip"] and rng.random() < 0.5:
        out = out[::-1]
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Leaf and proxy-lesion masks (PlantVillage-style uniform backgrounds)
# ---------------------------------------------------------------------------
def leaf_mask(img_rgb):
    """Binary leaf mask for uniform-background images: chroma distance (CIELAB a*,b*) from the
    background colour estimated on the image border, Otsu threshold, largest component, holes filled.
    Low-chroma shadows are therefore excluded and necrotic lesions inside the leaf are kept."""
    import cv2
    u8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    bw = 6
    border = np.concatenate([lab[:bw].reshape(-1, 3), lab[-bw:].reshape(-1, 3),
                             lab[:, :bw].reshape(-1, 3), lab[:, -bw:].reshape(-1, 3)])
    bg = np.median(border, 0)
    d = np.sqrt((lab[..., 1] - bg[1]) ** 2 + (lab[..., 2] - bg[2]) ** 2)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    d8 = np.clip(255 * d / (d.max() + 1e-6), 0, 255).astype(np.uint8)
    _, m = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        m = (lab_cc == k).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=-1)
    return filled > 0


def proxy_lesion_mask(img_rgb, leaf=None):
    """Automatic, colour-based proxy of symptomatic tissue (non-green pixels inside the leaf).
    This is NOT an expert annotation and must be reported as a proxy."""
    import cv2
    leaf = leaf_mask(img_rgb) if leaf is None else leaf
    u8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    r, g, b = [u8[..., i].astype(np.float32) for i in range(3)]
    exg = 2 * g - r - b
    nongreen = (h < 30) | (h > 90) | (exg < 10) | (v < 60)
    m = (nongreen & leaf).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # remove the thin leaf rim (edge pixels are often non-green after resizing)
    rim = cv2.erode(leaf.astype(np.uint8), np.ones((9, 9), np.uint8))
    return (m > 0) & (rim > 0)


# ---------------------------------------------------------------------------
# Corruptions for robustness analysis (3 severities each)
# ---------------------------------------------------------------------------
def corrupt(img, kind, sev, rng):
    import cv2
    x = img.astype(np.float32)
    if kind == "gaussian_noise":
        x = x + rng.normal(0, [10, 20, 30][sev - 1], x.shape)
    elif kind == "gaussian_blur":
        x = cv2.GaussianBlur(x, (0, 0), [1.0, 2.0, 3.0][sev - 1])
    elif kind == "motion_blur":
        k = [5, 9, 15][sev - 1]
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        x = cv2.filter2D(x, -1, ker)
    elif kind == "darken":
        x = x * [0.7, 0.5, 0.3][sev - 1]
    elif kind == "brighten":
        x = x * [1.3, 1.6, 1.9][sev - 1]
    elif kind == "low_contrast":
        f = [0.7, 0.5, 0.3][sev - 1]
        x = (x - x.mean()) * f + x.mean()
    elif kind == "jpeg":
        q = [50, 25, 10][sev - 1]
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, q])
        x = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB).astype(np.float32)
    elif kind == "occlusion":
        frac = [0.10, 0.20, 0.30][sev - 1]
        m = leaf_mask(img)
        ys, xs = np.where(m)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            area = frac * (y1 - y0 + 1) * (x1 - x0 + 1)
            side = int(np.sqrt(area))
            cy = rng.integers(y0, max(y0 + 1, y1 - side + 1))
            cx = rng.integers(x0, max(x0 + 1, x1 - side + 1))
            x[cy:cy + side, cx:cx + side] = 0.0
    elif kind == "background_clutter":
        m = leaf_mask(img)[..., None]
        h, w = x.shape[:2]
        base = np.array([[110, 85, 60], [70, 100, 50], [130, 120, 100]][rng.integers(0, 3)], np.float32)
        lum = cv2.resize(rng.normal(0, 1, (h // 8, w // 8)).astype(np.float32), (w, h),
                         interpolation=cv2.INTER_CUBIC)[..., None]
        chroma = cv2.resize(rng.normal(0, 1, (h // 16, w // 16, 3)).astype(np.float32), (w, h),
                            interpolation=cv2.INTER_CUBIC)
        amp = [20, 40, 60][sev - 1]
        bg = base + amp * lum + 0.25 * amp * chroma        # soil-like texture
        # random leaf-like green blobs at higher severities
        for _ in range([0, 4, 10][sev - 1]):
            cxy = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            ax = (int(rng.integers(10, 40)), int(rng.integers(5, 20)))
            cv2.ellipse(bg, cxy, ax, float(rng.uniform(0, 180)), 0, 360,
                        (float(rng.uniform(40, 90)), float(rng.uniform(100, 160)), float(rng.uniform(30, 70))), -1)
        x = np.where(m, x, bg)
    else:
        raise ValueError(kind)
    return np.clip(x, 0, 255).astype(np.float32)


CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "motion_blur", "darken", "brighten",
               "low_contrast", "jpeg", "occlusion", "background_clutter"]
