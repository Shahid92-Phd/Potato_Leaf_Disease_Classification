# MRV-Net: explainable multi-backbone transfer learning for potato leaf disease classification

Code accompanying the article *"MRV-Net: An Explainable Multi-Backbone Transfer Learning Framework
for Potato Leaf Disease Classification using Plant Image Dataset"* (S. M. Ganie).

Repository: https://github.com/Shahid92-Phd/Potato_Leaf_Disease_Classification

MRV-Net fuses the final feature maps of three frozen ImageNet backbones, **M**obileNetV2,
**R**esNet50 and **V**GG16. The pipeline in this repository trains MRV-Net and eight
single-backbone baselines under one protocol. It then produces every analysis reported in the
article: ablation, statistical tests, inference cost, robustness to image corruptions,
quantitative explainability and external validation on field images.

## Architecture

| Stage | Output shape |
|---|---|
| Input RGB image (256×256 on disk, bilinear resize) | 224 × 224 × 3 |
| Per-branch Keras `preprocess_input` (inside the model) | 224 × 224 × 3 (×3) |
| MobileNetV2 `out_relu` / ResNet50 `conv5_block3_out` / VGG16 `block5_pool` (frozen) | 7×7×1280 / 7×7×2048 / 7×7×512 |
| Concatenate → BatchNorm | 7 × 7 × 3840 |
| Conv2D 1024, 2×2, stride 1, `same`, ReLU | 7 × 7 × 1024 |
| MaxPool 2×2 → Flatten → Dropout 0.3 | 9216 |
| Dense 1024 → 512 → 256 (ReLU, dropout 0.3) → Dense 3 (Softmax) | 3 |

There are 66,400,515 parameters in total (25,832,451 trainable). The model has 40.64 GFLOPs per
224×224 image. `results/model_summary.txt` holds the full code-generated `model.summary()`.

## Training protocol (all nine models)

- **Data:** the published Kaggle split (900 / 300 / 300; 300 / 100 / 100 per class), used unchanged.
- **Backbones:** frozen ImageNet backbones, each with its own Keras preprocessing.
- **Optimisation:** categorical cross-entropy, Adam, batch 32.
- **Learning rate:** chosen per model from {1e-3, 1e-4} by the lowest validation loss.
- **Stopping:** at most 100 epochs, with early stopping on validation loss (patience 15, best
  weights restored).
- **Augmentation (training only, on the fly):** rotation 10°, width/height shift 0.2, zoom 0.2,
  shear 0.2, horizontal flip.
- **Seed:** 42.
- **Model selection:** the test set and the external dataset are never used for training or model
  selection.

## Repository layout

```
config.py                    all paths and settings (edit this file only)
common.py                    model builders, data loading, metrics, masks, corruptions
s00_audit_split.py           split export, exact/near-duplicate audit across splits
s00b_train_models.py         trains MRV-Net and the 8 baselines (learning-rate selection, early stopping)
s01_model_summary.py         model.summary(), parameters, internal test accuracy and confusion matrices
s02_cache_features.py        caches frozen-backbone features (original + 5 augmented copies per image)
s03_ablation.py              branch / fusion-component ablation and ensembles, 5 seeds
s04_statistics.py            McNemar (exact, Holm), Wilson/bootstrap CIs, seed-level tests, Friedman
s05_efficiency.py            GFLOPs, CPU latency, GPU throughput and memory (--gpu-torch subprocess)
s06_robustness.py            9 corruptions x 3 severities
s07_xai_quant.py             Grad-CAM (fusion layer and per branch), SHAP, deletion/insertion, stability
s08_external_validation.py   inference on the PLD field dataset (no retraining)
s08b_external_statistics.py  paired McNemar (Holm) and bootstrap CIs, MRV-Net vs each baseline on PLD
s09_learning_curves.py       training/validation accuracy and loss curves
s10_report.py                collects everything into results/REPORT.md
data/README.md               where to download and place the datasets
results/                     all outputs reported in the article (CSV/JSON/figures/predictions)
```

## Environment

The code was tested on Windows 11 with Python 3.12, Keras 3.15.1, TensorFlow 2.21.0 and
PyTorch 2.11.0+cu128, on an NVIDIA GeForce RTX 5080 Laptop GPU.

TensorFlow ≥ 2.11 has no native-Windows GPU support. When `torch.cuda.is_available()`, the
GPU-heavy scripts (`s00b`, `s02`, `s03` and the GPU timing in `s05`) therefore select the
**PyTorch backend of Keras 3** automatically. Models saved by one backend load in the other.

```bash
conda create -n mrvnet python=3.12 -y && conda activate mrvnet
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU build; adjust CUDA version
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

## Data

See `data/README.md` for the folder layout and download sources:

- **Training, validation and internal test:** the Kaggle Potato Leaf Disease dataset (Putra, 2020).
- **External test only:** the PLD field dataset (Rashid et al., 2021).

Paths are set in `config.py` (`DATA_ROOT`, `EXTERNAL_SETS`).

## Reproducing the results

Run from the repository root, in this order:

```bash
python s00_audit_split.py                 # ~1 min
python s00b_train_models.py               # ~3 h on one GPU (NASNetLarge dominates); restartable
python s01_model_summary.py
python s02_cache_features.py
python s03_ablation.py                    # 5 seeds x all configurations
python s04_statistics.py
python s05_efficiency.py                  # close other heavy programs first
python s06_robustness.py
python s07_xai_quant.py --shap
python s08_external_validation.py
python s08b_external_statistics.py       # add --set PLD_test for the 405-image test split
python s09_learning_curves.py
python s10_report.py
```

Useful options:

- `s00b_train_models.py --models MRV-Net ResNet50` trains a subset; `--overwrite` retrains models
  that already exist.
- `s03_ablation.py --only R+V --seeds 0 1` runs selected ablation configurations or seeds.
- `s07_xai_quant.py --limit 50` restricts the explainability analysis to a quick subset.

Trained weights (≈ 8 GB) and cached features are not stored in the repository. They are regenerated
by `s00b` and `s02`.

## Main results (from `results/`)

| Model | Internal test acc. (95% Wilson CI), n = 300 | Mean acc. under corruption | PLD field images, acc. (macro-F1), n = 4,072 |
|---|---|---|---|
| **MRV-Net** | **98.33% (96.2–99.3)** | **90.1%** | **65.3% (0.653)** |
| VGG16 | 96.67% | 87.7% | 61.9% (0.584) |
| ResNet50 | 99.00% | 87.8% | 61.0% (0.605) |
| MobileNetV2 | 96.67% | 86.6% | 60.2% (0.584) |
| Xception | 95.33% | 86.8% | 65.3% (0.632) |
| DenseNet201 | 99.00% | 92.6% | 58.3% (0.560) |
| EfficientNetB0 | 100.00% | 95.0% | 56.4% (0.519) |
| NASNetLarge | 96.67% | 82.6% | 46.5% (0.418) |
| DenseNet121 | 98.33% | 91.0% | 55.3% (0.500) |

- **Internal test set:** no pairwise difference is significant after Holm correction (exact
  McNemar).
- **PLD field images:** MRV-Net is significantly more accurate than each of its three backbones
  and than the other baselines (Holm-adjusted p < 0.001). The exception is Xception, with which it
  is statistically equivalent. On the smaller 405-image PLD test split, only the differences from
  EfficientNetB0, DenseNet121 and NASNetLarge remain significant
  (`results/stats_mcnemar_external_PLD_test.csv`).
- **Detailed outputs:** `results/REPORT.md`; the ablation, explainability and efficiency tables
  are in the corresponding CSV files.

## Citation

Please cite the article (details will be added on publication) and this repository
(`CITATION.cff`).

## Licence

MIT (see `LICENSE`). The datasets remain under their original licences.
