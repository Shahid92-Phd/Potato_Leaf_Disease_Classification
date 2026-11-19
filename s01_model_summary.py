"""
S01 - Code-generated model.summary() and architecture verification  (R1.1-R1.4, R1.6, R2.2)

Outputs
  results/model_summary.txt          full Keras summary (top level + expanded nested backbones)
  results/model_layers.csv           every top-level layer: type, output shape, params, trainable, config
  results/s01_summary.json           totals, trainable / non-trainable, detected branch order,
                                     fusion-conv kernel/stride/padding/activation, backbone layers
  results/predictions/MRV-Net_saved.csv   per-image test probabilities of the SAVED model
"""
import io
import os
import contextlib

import numpy as np
import pandas as pd

import config as C
from common import (ensure_dirs, load_model, keras, save_json, load_split, load_subset,
                    predict_probs, save_preds, metrics_from_preds, wilson_ci)


def count(ws):
    return int(sum(np.prod(w.shape) for w in ws))


def out_shape(layer):
    try:
        return str(layer.output_shape)
    except Exception:
        try:
            return str(tuple(layer.output.shape))
        except Exception:
            return "?"


def main():
    ensure_dirs()
    m = load_model(C.MRVNET_PATH)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.summary(line_length=120, expand_nested=False)
        print("\n\n===== Expanded (nested backbones) =====\n")
        try:
            m.summary(line_length=120, expand_nested=True)
        except Exception as e:
            print("expand_nested not supported:", e)
    txt = buf.getvalue()
    with open(os.path.join(C.RESULTS_DIR, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt.split("===== Expanded")[0])

    rows = []
    for l in m.layers:
        cfg = l.get_config() if not isinstance(l, keras.Model) else {"name": l.name}
        keep = {k: cfg.get(k) for k in ("filters", "kernel_size", "strides", "padding", "activation",
                                        "pool_size", "rate", "units", "axis") if k in cfg}
        rows.append(dict(name=l.name, type=type(l).__name__, output_shape=out_shape(l),
                         params=l.count_params(), trainable_params=count(l.trainable_weights),
                         non_trainable_params=count(l.non_trainable_weights),
                         layer_trainable=bool(l.trainable), config=str(keep)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS_DIR, "model_layers.csv"), index=False)

    convs = [l for l in m.layers if isinstance(l, keras.layers.Conv2D)]
    concat = [l for l in m.layers if isinstance(l, keras.layers.Concatenate)]
    subs = [l for l in m.layers if isinstance(l, keras.Model)]
    order = []
    if concat:
        try:
            inb = concat[0]._inbound_nodes[-1]
            srcs = getattr(inb, "inbound_layers", None) or [t._keras_history[0] for t in inb.input_tensors]
            order = [s.name for s in (srcs if isinstance(srcs, list) else [srcs])]
        except Exception:
            order = []
    lambdas = [l.name for l in m.layers if isinstance(l, keras.layers.Lambda)]

    summary = dict(
        total_params=int(m.count_params()),
        trainable_params=count(m.trainable_weights),
        non_trainable_params=count(m.non_trainable_weights),
        weights_MB_float32=round(m.count_params() * 4 / 2 ** 20, 2),
        input_shape=str(m.input_shape),
        nested_backbones=[dict(name=s.name, output_shape=out_shape(s), params=s.count_params(),
                               trainable=bool(s.trainable),
                               trainable_params=count(s.trainable_weights),
                               last_layer=s.layers[-1].name) for s in subs],
        concat_input_order=order,
        concat_output_shape=out_shape(concat[0]) if concat else None,
        preprocessing_lambda_layers=lambdas,
        fusion_conv=[dict(name=c.name, filters=c.filters, kernel_size=c.kernel_size, strides=c.strides,
                          padding=c.padding, activation=keras.activations.serialize(c.activation)
                          if not isinstance(c.activation, str) else c.activation,
                          output_shape=out_shape(c), params=c.count_params()) for c in convs],
    )
    # Test-set predictions of the saved model (used by S04)
    try:
        split = load_split()
        x, y, d = load_subset(split, "test")
        probs = predict_probs(m, x, C.MRVNET_INPUT_MODE)
        save_preds("MRV-Net_saved", d, probs)
        met = metrics_from_preds(y, probs.argmax(1))
        k = int((probs.argmax(1) == y).sum())
        lo, hi = wilson_ci(k, len(y))
        summary["saved_model_test_accuracy"] = met["accuracy"]
        summary["saved_model_test_accuracy_wilson95"] = [lo, hi]
        summary["saved_model_confusion_matrix"] = met["confusion_matrix"]
        print(f"\nMRV-Net test accuracy (retrained model): {met['accuracy']:.4f}")
        print("Confusion matrix:", met["confusion_matrix"])
    except FileNotFoundError as e:
        print("Skipping test predictions:", e)

    save_json(summary, os.path.join(C.RESULTS_DIR, "s01_summary.json"))
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
