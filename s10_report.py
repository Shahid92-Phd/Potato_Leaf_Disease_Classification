"""
S10 - Collect every result into results/REPORT.md
"""
import os
import json

import pandas as pd

import config as C

R = C.RESULTS_DIR


def md(df, floatfmt=4):
    if df is None or len(df) == 0:
        return "_not available_\n"
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in df.values)


def rd(name, **kw):
    p = os.path.join(R, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def rj(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    out = ["# MRV-Net - results report\n",
           "Summary of all analyses reported in the article.\n"]

    out.append("\n## Training of all models under one protocol (s00b)\n")
    out.append(md(rd("training_log.csv")))
    p = rj("s00b_protocol.json")
    out.append("\n```\n" + json.dumps(p, indent=2) + "\n```\n" if p else "")

    out.append("\n## Split and duplicate audit (s00)\n")
    out.append(md(rd("split_counts.csv")))
    s = rj("s00_summary.json")
    out.append("\n```\n" + json.dumps(s, indent=2) + "\n```\n" if s else "_not run_\n")

    out.append("\n## Code-generated model summary (s01)\n")
    s = rj("s01_summary.json")
    if s:
        keep = {k: s.get(k) for k in ("total_params", "trainable_params", "non_trainable_params", "weights_MB_float32",
                                      "input_shape", "concat_output_shape", "concat_input_order",
                                      "preprocessing_lambda_layers", "fusion_conv", "nested_backbones",
                                      "saved_model_test_accuracy", "saved_model_test_accuracy_wilson95",
                                      "saved_model_confusion_matrix")}
        out.append("```\n" + json.dumps(keep, indent=2) + "\n```\n")
        out.append("Full text: results/model_summary.txt (paste into Supplementary File S1)\n")

    ab = rd("ablation_summary.csv")
    cols = ["config", "test_accuracy_mean", "test_accuracy_std", "test_macro_precision_mean",
            "test_macro_recall_mean", "test_macro_f1_mean", "test_macro_f1_std", "dedup_test_accuracy_mean",
            "head_params", "train_seconds_mean", "n_seeds"]
    if ab is not None:
        ab = ab[[c for c in cols if c in ab.columns]]
        out.append("\n## Branch ablation (mean, SD over seeds) (s03)\n")
        out.append(md(ab[ab.config.isin(["M", "R", "V", "M+R", "M+V", "R+V", "M+R+V"])]))
        out.append("\n## Fusion-component ablation (s03)\n")
        out.append(md(ab[ab.config.str.contains(r"\|") | (ab.config == "M+R+V")]))
        out.append("\n## Prediction-level ensembles vs feature-level fusion (s03)\n")
        out.append(md(ab[ab.config.str.startswith("Ensemble") | (ab.config == "M+R+V")]))
    else:
        out.append("\n## Ablation\n_not run_\n")

    out.append("\n## Sample-level statistics (s04)\n")
    out.append("### 95% confidence intervals\n" + md(rd("stats_model_CIs_full.csv")))
    out.append("\n### Exact McNemar, Holm-adjusted\n" + md(rd("stats_mcnemar_full.csv")))
    out.append("\n### Ablation variants across independent runs\n" + md(rd("stats_ablation_seeds.csv")))
    f = rj("stats_friedman_aligned_reproduction.json")
    if f:
        out.append("\n### Reproduction of original Friedman aligned ranks\n```\n" + json.dumps(f, indent=2) + "\n```\n")
        out.append(md(rd("stats_friedman_aligned_reproduction.csv"), 5))

    out.append("\n## Computational cost (s05)\n")
    out.append(md(rd("efficiency.csv"), 2))
    e = rj("efficiency_env.json")
    if e:
        out.append(f"\nMeasured on: {e}\n")

    out.append("\n## Corruption robustness (s06)\n")
    out.append(md(rd("robustness_summary.csv")))
    out.append("\n" + md(rd("robustness_by_corruption.csv")))

    out.append("\n## Quantitative XAI (s07)\n")
    x = rd("xai_summary.csv")
    if x is not None:
        x = x.rename(columns={x.columns[0]: "subset"})
        keep = [c for c in ["subset", "n", "leaf_area", "leaf_energy", "proxy_lesion_energy", "proxy_pointing_hit",
                            "proxy_iou_top20", "expert_lesion_energy", "expert_pointing_hit", "expert_iou_top20",
                            "deletion_auc_sal", "deletion_auc_rand", "insertion_auc_sal", "insertion_auc_rand",
                            "stability_noise", "stability_brightness", "stability_shift", "shap_leaf_energy",
                            "shap_gradcam_spearman", "leaf_energy_MobileNetV2", "leaf_energy_ResNet50",
                            "leaf_energy_VGG16"] if c in x.columns]
        out.append(md(x[keep], 3))
    out.append("\n" + md(rd("xai_tests.csv"), 6))

    out.append("\n## External validation (s08)\n")
    out.append(md(rd("external_validation.csv")))

    out.append("\n## De-duplicated test subset (s04)\n")
    out.append(md(rd("stats_model_CIs_dedup.csv")))
    out.append("\n" + md(rd("stats_mcnemar_dedup.csv")))

    with open(os.path.join(R, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    print("Wrote", os.path.join(R, "REPORT.md"))


if __name__ == "__main__":
    main()
