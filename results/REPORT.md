# MRV-Net - results report
Summary of all analyses reported in the article.

## Training of all models under one protocol (s00b)
| model | selected_lr | epochs_run | best_epoch | best_val_loss | best_val_accuracy | train_seconds | total_params | device | seed | val_loss_lr0.001 | val_acc_lr0.001 | val_loss_lr0.0001 | val_acc_lr0.0001 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MRV-Net | 0.0001 | 26 | 11 | 0.0105 | 0.9967 | 170.5000 | 66400515 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 10.7454 | 0.3333 | 0.0105 | 0.9967 |
| VGG16 | 0.0010 | 38 | 23 | 0.0534 | 0.9833 | 103.7000 | 20093251 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 0.0534 | 0.9833 | 0.0647 | 0.9767 |
| ResNet50 | 0.0001 | 70 | 55 | 0.0103 | 0.9933 | 166.1000 | 43128195 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 0.3646 | 0.9633 | 0.0103 | 0.9933 |
| MobileNetV2 | 0.0001 | 44 | 29 | 0.0471 | 0.9767 | 76.8000 | 14717507 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 0.0768 | 0.9733 | 0.0471 | 0.9767 |
| Xception | 0.0001 | 55 | 40 | 0.0769 | 0.9733 | 157.9000 | 40401963 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 0.0934 | 0.9467 | 0.0769 | 0.9733 |
| DenseNet201 | 0.0010 | 59 | 44 | 0.0139 | 0.9933 | 312.9000 | 36682307 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 0.0139 | 0.9933 | 0.0365 | 0.9867 |
| EfficientNetB0 | 0.0001 | 60 | 45 | 0.0060 | 0.9967 | 180.1000 | 16509094 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 0.0158 | 1.0000 | 0.0060 | 0.9967 |
| NASNetLarge | 0.0001 | 41 | 26 | 0.1736 | 0.9433 | 7631.8000 | 122749781 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 2.2341 | 0.8467 | 0.1736 | 0.9433 |
| DenseNet121 | 0.0001 | 81 | 66 | 0.0084 | 0.9933 | 281.8000 | 17136707 | NVIDIA GeForce RTX 5080 Laptop GPU (PyTorch backend) | 42 | 0.0101 | 0.9933 | 0.0084 | 0.9933 |

```
{
  "models": [
    "MRV-Net",
    "VGG16",
    "ResNet50",
    "MobileNetV2",
    "Xception",
    "DenseNet201",
    "EfficientNetB0",
    "NASNetLarge",
    "DenseNet121"
  ],
  "seed": 42,
  "augment": {
    "rotation": 10,
    "shear": 0.2,
    "zoom": 0.2,
    "shift": 0.2,
    "hflip": true,
    "vflip": false
  },
  "img_size": 224,
  "max_epochs": 100,
  "patience": 15,
  "lr_grid": [
    0.001,
    0.0001
  ],
  "batch": 32
}
```

## Split and duplicate audit (s00)
| split | Early blight | Late blight | Healthy | Total |
|---|---|---|---|---|
| test | 100 | 100 | 100 | 300 |
| train | 300 | 300 | 300 | 900 |
| val | 100 | 100 | 100 | 300 |

```
{
  "n_images": 1500,
  "unique_sizes": [
    [
      256,
      256
    ]
  ],
  "n_exact_duplicate_files": 0,
  "n_near_duplicate_pairs": 54,
  "n_near_duplicate_pairs_by_class": {
    "Early blight": 8,
    "Late blight": 20,
    "Healthy": 26
  },
  "n_cross_split_pairs": 14,
  "n_label_conflicts": 15,
  "n_test_images": 300,
  "n_test_with_train_or_val_near_duplicate": 6,
  "n_test_dedup": 294,
  "hamming_threshold": 6
}
```

## Code-generated model summary (s01)
```
{
  "total_params": 66400515,
  "trainable_params": 25832451,
  "non_trainable_params": 40568064,
  "weights_MB_float32": 253.3,
  "input_shape": "(None, 224, 224, 3)",
  "concat_output_shape": "(None, 7, 7, 3840)",
  "concat_input_order": [
    "mobilenetv2_1.00_224",
    "resnet50",
    "vgg16"
  ],
  "preprocessing_lambda_layers": [],
  "fusion_conv": [
    {
      "name": "conv2d",
      "filters": 1024,
      "kernel_size": [
        2,
        2
      ],
      "strides": [
        1,
        1
      ],
      "padding": "same",
      "activation": "relu",
      "output_shape": "(None, 7, 7, 1024)",
      "params": 15729664
    }
  ],
  "nested_backbones": [
    {
      "name": "mobilenetv2_1.00_224",
      "output_shape": "(None, 7, 7, 1280)",
      "params": 2257984,
      "trainable": false,
      "trainable_params": 0,
      "last_layer": "out_relu"
    },
    {
      "name": "resnet50",
      "output_shape": "(None, 7, 7, 2048)",
      "params": 23587712,
      "trainable": false,
      "trainable_params": 0,
      "last_layer": "conv5_block3_out"
    },
    {
      "name": "vgg16",
      "output_shape": "(None, 7, 7, 512)",
      "params": 14714688,
      "trainable": false,
      "trainable_params": 0,
      "last_layer": "block5_pool"
    }
  ],
  "saved_model_test_accuracy": 0.9833333333333333,
  "saved_model_test_accuracy_wilson95": [
    0.9615846049313345,
    0.9928605226945714
  ],
  "saved_model_confusion_matrix": [
    [
      100,
      0,
      0
    ],
    [
      0,
      97,
      3
    ],
    [
      0,
      2,
      98
    ]
  ]
}
```
Full text: results/model_summary.txt (paste into Supplementary File S1)

## Branch ablation (mean, SD over seeds) (s03)
| config | test_accuracy_mean | test_accuracy_std | test_macro_precision_mean | test_macro_recall_mean | test_macro_f1_mean | test_macro_f1_std | head_params | train_seconds_mean | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| M | 0.9747 | 0.0056 | 0.9749 | 0.9747 | 0.9747 | 0.0056 | 15344131.0000 | 43.4600 | 5 |
| R | 0.9880 | 0.0051 | 0.9883 | 0.9880 | 0.9880 | 0.0050 | 18492931.0000 | 85.2600 | 5 |
| V | 0.9780 | 0.0051 | 0.9783 | 0.9780 | 0.9780 | 0.0051 | 12195331.0000 | 40.8200 | 5 |
| M+R | 0.9880 | 0.0051 | 0.9882 | 0.9880 | 0.9880 | 0.0050 | 23740931.0000 | 107.0200 | 5 |
| M+V | 0.9827 | 0.0068 | 0.9828 | 0.9827 | 0.9827 | 0.0068 | 17443331.0000 | 61.3200 | 5 |
| R+V | 0.9940 | 0.0028 | 0.9941 | 0.9940 | 0.9940 | 0.0028 | 20592131.0000 | 140.2800 | 5 |
| M+R+V | 0.9880 | 0.0018 | 0.9881 | 0.9880 | 0.9880 | 0.0018 | 25840131.0000 | 129.7800 | 5 |

## Fusion-component ablation (s03)
| config | test_accuracy_mean | test_accuracy_std | test_macro_precision_mean | test_macro_recall_mean | test_macro_f1_mean | test_macro_f1_std | head_params | train_seconds_mean | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| M+R+V | 0.9880 | 0.0018 | 0.9881 | 0.9880 | 0.9880 | 0.0018 | 25840131.0000 | 129.7800 | 5 |
| M+R+V|no_bn | 0.9880 | 0.0045 | 0.9882 | 0.9880 | 0.9880 | 0.0045 | 25824771.0000 | 145.7800 | 5 |
| M+R+V|conv1x1 | 0.9893 | 0.0028 | 0.9894 | 0.9893 | 0.9893 | 0.0028 | 14043651.0000 | 130.1000 | 5 |
| M+R+V|no_conv | 0.9880 | 0.0051 | 0.9883 | 0.9880 | 0.9880 | 0.0051 | 36062723.0000 | 119.5000 | 5 |
| M+R+V|gap_head | 0.9907 | 0.0043 | 0.9907 | 0.9907 | 0.9907 | 0.0043 | 17451523.0000 | 119.3200 | 5 |
| M+R+V|no_dropout | 0.9853 | 0.0038 | 0.9856 | 0.9853 | 0.9853 | 0.0038 | 25840131.0000 | 167.1400 | 5 |

## Prediction-level ensembles vs feature-level fusion (s03)
| config | test_accuracy_mean | test_accuracy_std | test_macro_precision_mean | test_macro_recall_mean | test_macro_f1_mean | test_macro_f1_std | head_params | train_seconds_mean | n_seeds |
|---|---|---|---|---|---|---|---|---|---|
| M+R+V | 0.9880 | 0.0018 | 0.9881 | 0.9880 | 0.9880 | 0.0018 | 25840131.0000 | 129.7800 | 5 |
| Ensemble soft-vote (M,R,V) | 0.9953 | 0.0030 | 0.9954 | 0.9953 | 0.9953 | 0.0030 |  |  | 5 |
| Ensemble hard-vote (M,R,V) | 0.9953 | 0.0030 | 0.9954 | 0.9953 | 0.9953 | 0.0030 |  |  | 5 |

## Sample-level statistics (s04)
### 95% confidence intervals
| model | n | accuracy | wilson_lo | wilson_hi | boot_acc_lo | boot_acc_hi | macro_f1 | boot_f1_lo | boot_f1_hi |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNetB0 | 300 | 1.0000 | 0.9874 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DenseNet201 | 300 | 0.9900 | 0.9710 | 0.9966 | 0.9767 | 1.0000 | 0.9900 | 0.9771 | 1.0000 |
| ResNet50 | 300 | 0.9900 | 0.9710 | 0.9966 | 0.9767 | 1.0000 | 0.9900 | 0.9769 | 1.0000 |
| MRV-Net | 300 | 0.9833 | 0.9616 | 0.9929 | 0.9667 | 0.9967 | 0.9833 | 0.9677 | 0.9966 |
| DenseNet121 | 300 | 0.9833 | 0.9616 | 0.9929 | 0.9700 | 0.9967 | 0.9834 | 0.9669 | 0.9966 |
| VGG16 | 300 | 0.9667 | 0.9397 | 0.9818 | 0.9433 | 0.9867 | 0.9664 | 0.9445 | 0.9864 |
| MobileNetV2 | 300 | 0.9667 | 0.9397 | 0.9818 | 0.9433 | 0.9833 | 0.9668 | 0.9455 | 0.9866 |
| NASNetLarge | 300 | 0.9667 | 0.9397 | 0.9818 | 0.9433 | 0.9867 | 0.9666 | 0.9457 | 0.9853 |
| Xception | 300 | 0.9533 | 0.9232 | 0.9720 | 0.9267 | 0.9767 | 0.9531 | 0.9289 | 0.9739 |

### Exact McNemar, Holm-adjusted
| comparison | mrv_only_correct | other_only_correct | discordant | p_exact | acc_diff | acc_diff_lo | acc_diff_hi | f1_diff | f1_diff_lo | f1_diff_hi | p_holm | significant_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MRV-Net vs Xception | 12 | 3 | 15 | 0.0352 | 0.0300 | 0.0067 | 0.0533 | 0.0302 | 0.0065 | 0.0558 | 0.2812 | No |
| MRV-Net vs EfficientNetB0 | 0 | 5 | 5 | 0.0625 | -0.0167 | -0.0333 | -0.0033 | -0.0167 | -0.0322 | -0.0034 | 0.4375 | No |
| MRV-Net vs MobileNetV2 | 8 | 3 | 11 | 0.2266 | 0.0167 | -0.0033 | 0.0367 | 0.0166 | -0.0037 | 0.0385 | 1.0000 | No |
| MRV-Net vs VGG16 | 8 | 3 | 11 | 0.2266 | 0.0167 | -0.0033 | 0.0400 | 0.0170 | -0.0037 | 0.0409 | 1.0000 | No |
| MRV-Net vs NASNetLarge | 9 | 4 | 13 | 0.2668 | 0.0167 | -0.0067 | 0.0400 | 0.0167 | -0.0071 | 0.0406 | 1.0000 | No |
| MRV-Net vs ResNet50 | 2 | 4 | 6 | 0.6875 | -0.0067 | -0.0233 | 0.0100 | -0.0067 | -0.0237 | 0.0098 | 1.0000 | No |
| MRV-Net vs DenseNet201 | 2 | 4 | 6 | 0.6875 | -0.0067 | -0.0233 | 0.0100 | -0.0067 | -0.0235 | 0.0092 | 1.0000 | No |
| MRV-Net vs DenseNet121 | 4 | 4 | 8 | 1.0000 | 0.0000 | -0.0167 | 0.0167 | -0.0000 | -0.0181 | 0.0175 | 1.0000 | No |

### Ablation variants across independent runs
| metric | comparison | n_seeds | mean_ref | mean_other | mean_diff | ci_lo | ci_hi | p_ttest | p_wilcoxon | p_ttest_holm | p_wilcoxon_holm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test_accuracy | M+R+V vs Ensemble hard-vote (M,R,V) | 5 | 0.9880 | 0.9953 | -0.0073 | -0.0127 | -0.0019 | 0.0196 | 0.1250 | 0.2151 | 1.0000 |
| test_accuracy | M+R+V vs Ensemble soft-vote (M,R,V) | 5 | 0.9880 | 0.9953 | -0.0073 | -0.0127 | -0.0019 | 0.0196 | 0.1250 | 0.2151 | 1.0000 |
| test_accuracy | M+R+V vs M | 5 | 0.9880 | 0.9747 | 0.0133 | 0.0083 | 0.0184 | 0.0019 | 0.0625 | 0.0243 | 0.8125 |
| test_accuracy | M+R+V vs M+R | 5 | 0.9880 | 0.9880 | 0.0000 | -0.0083 | 0.0083 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test_accuracy | M+R+V vs M+R+V|conv1x1 | 5 | 0.9880 | 0.9893 | -0.0013 | -0.0069 | 0.0042 | 0.5415 | 0.7500 | 1.0000 | 1.0000 |
| test_accuracy | M+R+V vs M+R+V|gap_head | 5 | 0.9880 | 0.9907 | -0.0027 | -0.0088 | 0.0035 | 0.2943 | 0.5000 | 1.0000 | 1.0000 |
| test_accuracy | M+R+V vs M+R+V|no_bn | 5 | 0.9880 | 0.9880 | 0.0000 | -0.0041 | 0.0041 | 1.0000 | 0.5000 | 1.0000 | 1.0000 |
| test_accuracy | M+R+V vs M+R+V|no_conv | 5 | 0.9880 | 0.9880 | 0.0000 | -0.0065 | 0.0065 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test_accuracy | M+R+V vs M+R+V|no_dropout | 5 | 0.9880 | 0.9853 | 0.0027 | -0.0008 | 0.0061 | 0.0993 | 0.2500 | 0.7944 | 1.0000 |
| test_accuracy | M+R+V vs M+V | 5 | 0.9880 | 0.9827 | 0.0053 | -0.0042 | 0.0149 | 0.1951 | 0.1875 | 1.0000 | 1.0000 |
| test_accuracy | M+R+V vs R | 5 | 0.9880 | 0.9880 | 0.0000 | -0.0065 | 0.0065 | 1.0000 | 0.8750 | 1.0000 | 1.0000 |
| test_accuracy | M+R+V vs R+V | 5 | 0.9880 | 0.9940 | -0.0060 | -0.0114 | -0.0006 | 0.0367 | 0.1250 | 0.3301 | 1.0000 |
| test_accuracy | M+R+V vs V | 5 | 0.9880 | 0.9780 | 0.0100 | 0.0049 | 0.0151 | 0.0054 | 0.0625 | 0.0649 | 0.8125 |
| test_accuracy | Friedman over branch configs (seeds=blocks) | 5 |  |  | 20.4312 |  |  | 0.0023 |  |  |  |
| test_macro_f1 | M+R+V vs Ensemble hard-vote (M,R,V) | 5 | 0.9880 | 0.9953 | -0.0073 | -0.0127 | -0.0019 | 0.0196 | 0.1250 | 0.2152 | 1.0000 |
| test_macro_f1 | M+R+V vs Ensemble soft-vote (M,R,V) | 5 | 0.9880 | 0.9953 | -0.0073 | -0.0127 | -0.0019 | 0.0196 | 0.1250 | 0.2152 | 1.0000 |
| test_macro_f1 | M+R+V vs M | 5 | 0.9880 | 0.9747 | 0.0133 | 0.0083 | 0.0184 | 0.0019 | 0.0625 | 0.0243 | 0.8125 |
| test_macro_f1 | M+R+V vs M+R | 5 | 0.9880 | 0.9880 | -0.0000 | -0.0083 | 0.0082 | 0.9934 | 1.0000 | 1.0000 | 1.0000 |
| test_macro_f1 | M+R+V vs M+R+V|conv1x1 | 5 | 0.9880 | 0.9893 | -0.0013 | -0.0069 | 0.0042 | 0.5398 | 0.3750 | 1.0000 | 1.0000 |
| test_macro_f1 | M+R+V vs M+R+V|gap_head | 5 | 0.9880 | 0.9907 | -0.0027 | -0.0088 | 0.0035 | 0.2933 | 0.3125 | 1.0000 | 1.0000 |
| test_macro_f1 | M+R+V vs M+R+V|no_bn | 5 | 0.9880 | 0.9880 | -0.0000 | -0.0041 | 0.0041 | 0.9975 | 0.8125 | 1.0000 | 1.0000 |
| test_macro_f1 | M+R+V vs M+R+V|no_conv | 5 | 0.9880 | 0.9880 | -0.0000 | -0.0065 | 0.0065 | 0.9990 | 1.0000 | 1.0000 | 1.0000 |
| test_macro_f1 | M+R+V vs M+R+V|no_dropout | 5 | 0.9880 | 0.9853 | 0.0027 | -0.0008 | 0.0061 | 0.1012 | 0.3125 | 0.8094 | 1.0000 |
| test_macro_f1 | M+R+V vs M+V | 5 | 0.9880 | 0.9827 | 0.0053 | -0.0042 | 0.0149 | 0.1959 | 0.3125 | 1.0000 | 1.0000 |
| test_macro_f1 | M+R+V vs R | 5 | 0.9880 | 0.9880 | -0.0000 | -0.0065 | 0.0065 | 0.9942 | 0.8125 | 1.0000 | 1.0000 |
| test_macro_f1 | M+R+V vs R+V | 5 | 0.9880 | 0.9940 | -0.0060 | -0.0114 | -0.0006 | 0.0367 | 0.1250 | 0.3306 | 1.0000 |
| test_macro_f1 | M+R+V vs V | 5 | 0.9880 | 0.9780 | 0.0100 | 0.0049 | 0.0152 | 0.0056 | 0.0625 | 0.0672 | 0.8125 |
| test_macro_f1 | Friedman over branch configs (seeds=blocks) | 5 |  |  | 20.1818 |  |  | 0.0026 |  |  |  |

## Computational cost (s05)
| model | input | total_params | trainable_params | weights_MB | file_MB | GFLOPs | cpu_latency_ms_median | cpu_latency_ms_p95 | weights_on_gpu | gpu_latency_ms_median | gpu_latency_ms_p95 | gpu_batch32_ms_median | gpu_throughput_img_s | gpu_peak_memory_MB | gpu_name | gpu_backend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MRV-Net | 224x224x3 | 66400515 | 25832451 | 253.30 | 451.68 | 40.64 | 41.06 | 42.10 | True | 60.30 | 63.46 | 135.98 | 235.32 | 5015.50 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| VGG16 | 224x224x3 | 20093251 | 5377539 | 76.65 | 117.80 | 30.72 | 29.04 | 30.39 | True | 5.27 | 5.47 | 60.30 | 530.69 | 2462.15 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| ResNet50 | 224x224x3 | 43128195 | 19536387 | 164.52 | 314.24 | 7.79 | 19.58 | 20.27 | True | 29.87 | 31.02 | 49.08 | 652.04 | 5242.84 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| MobileNetV2 | 224x224x3 | 14717507 | 12456963 | 56.14 | 151.78 | 0.64 | 9.56 | 11.54 | True | 25.46 | 26.35 | 24.86 | 1287.21 | 2752.09 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| Xception | 224x224x3 | 40401963 | 19536387 | 154.12 | 303.69 | 9.17 | 20.96 | 21.51 | True | 24.04 | 25.07 | 61.92 | 516.76 | 4495.73 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| DenseNet201 | 224x224x3 | 36682307 | 18356483 | 139.93 | 282.42 | 8.67 | 42.64 | 44.21 | True | 110.31 | 113.91 | 107.32 | 298.17 | 9904.79 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| EfficientNetB0 | 224x224x3 | 16509094 | 12456963 | 62.98 | 158.86 | 0.83 | 14.31 | 15.02 | True | 37.33 | 38.78 | 36.86 | 868.13 | 3192.93 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| NASNetLarge | 224x224x3 | 122749781 | 37824899 | 468.25 | 760.43 | 20.74 | 37.80 | 41.78 | True | 173.21 | 179.17 | 1784.69 | 17.93 | 15978.05 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |
| DenseNet121 | 224x224x3 | 17136707 | 10097155 | 65.37 | 143.90 | 5.72 | 26.03 | 27.77 | True | 65.92 | 68.36 | 64.65 | 494.99 | 6214.95 | NVIDIA GeForce RTX 5080 Laptop GPU | keras-torch |

Measured on: {'gpu': 'CPU only', 'tf': '2.21.0', 'platform': 'Windows-11-10.0.26300-SP0', 'cpu': 'Intel64 Family 6 Model 198 Stepping 2, GenuineIntel'}

## Corruption robustness (s06)
| model | mCA | mean_macro_f1 | mean_relative_accuracy | worst_accuracy | clean_accuracy |
|---|---|---|---|---|---|
| DenseNet121 | 0.9098 | 0.9080 | 0.9252 | 0.6500 | 0.9833 |
| DenseNet201 | 0.9260 | 0.9241 | 0.9354 | 0.6300 | 0.9900 |
| EfficientNetB0 | 0.9495 | 0.9495 | 0.9495 | 0.8800 | 1.0000 |
| MRV-Net | 0.9014 | 0.8944 | 0.9166 | 0.6167 | 0.9833 |
| MobileNetV2 | 0.8662 | 0.8600 | 0.8960 | 0.6967 | 0.9667 |
| NASNetLarge | 0.8257 | 0.8257 | 0.8542 | 0.6433 | 0.9667 |
| ResNet50 | 0.8775 | 0.8673 | 0.8864 | 0.5000 | 0.9900 |
| VGG16 | 0.8770 | 0.8725 | 0.9073 | 0.5167 | 0.9667 |
| Xception | 0.8677 | 0.8608 | 0.9101 | 0.6367 | 0.9533 |

| corruption | DenseNet121 | DenseNet201 | EfficientNetB0 | MRV-Net | MobileNetV2 | NASNetLarge | ResNet50 | VGG16 | Xception |
|---|---|---|---|---|---|---|---|---|---|
| background_clutter | 0.8178 | 0.7233 | 0.8867 | 0.7500 | 0.7967 | 0.8000 | 0.8544 | 0.6678 | 0.7856 |
| brighten | 0.9011 | 0.9456 | 0.9900 | 0.9600 | 0.9633 | 0.9178 | 0.9511 | 0.9456 | 0.9100 |
| darken | 0.9578 | 0.9933 | 0.9467 | 0.9100 | 0.9289 | 0.8167 | 0.7189 | 0.9389 | 0.9367 |
| gaussian_blur | 0.9433 | 0.9433 | 0.9467 | 0.9667 | 0.9278 | 0.8000 | 0.9522 | 0.9644 | 0.9044 |
| gaussian_noise | 0.8244 | 0.8889 | 0.9511 | 0.7978 | 0.8233 | 0.7744 | 0.7922 | 0.9067 | 0.7867 |
| jpeg | 0.8944 | 0.9556 | 0.9656 | 0.9533 | 0.9167 | 0.8400 | 0.9656 | 0.8989 | 0.8578 |
| low_contrast | 0.9711 | 0.9856 | 0.9700 | 0.8444 | 0.8933 | 0.8933 | 0.7433 | 0.9478 | 0.9344 |
| motion_blur | 0.9578 | 0.9822 | 0.9578 | 0.9600 | 0.8011 | 0.7956 | 0.9544 | 0.9200 | 0.8967 |
| occlusion | 0.9200 | 0.9167 | 0.9311 | 0.9700 | 0.7444 | 0.7933 | 0.9656 | 0.7033 | 0.7967 |

## Quantitative XAI (s07)
| subset | n | leaf_area | leaf_energy | proxy_lesion_energy | proxy_pointing_hit | proxy_iou_top20 | deletion_auc_sal | deletion_auc_rand | insertion_auc_sal | insertion_auc_rand | stability_noise | stability_brightness | stability_shift | shap_leaf_energy | shap_gradcam_spearman | leaf_energy_MobileNetV2 | leaf_energy_ResNet50 | leaf_energy_VGG16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All | 300 | 0.491 | 0.677 | 0.197 | 0.798 | 0.155 | 0.810 | 0.624 | 0.928 | 0.622 | 0.981 | 0.992 | 0.926 | 0.607 | 0.148 | 0.712 | 0.691 | 0.607 |
| Early blight | 100 | 0.542 | 0.758 | 0.206 | 0.890 | 0.160 | 0.836 | 0.587 | 0.898 | 0.589 | 0.994 | 0.998 | 0.958 | 0.645 | 0.186 | 0.825 | 0.863 | 0.622 |
| Healthy | 100 | 0.491 | 0.627 |  |  |  | 0.812 | 0.361 | 0.914 | 0.357 | 0.963 | 0.984 | 0.879 | 0.526 | -0.064 | 0.643 | 0.602 | 0.595 |
| Late blight | 100 | 0.439 | 0.646 | 0.188 | 0.699 | 0.150 | 0.782 | 0.923 | 0.972 | 0.921 | 0.985 | 0.995 | 0.941 | 0.650 | 0.322 | 0.667 | 0.605 | 0.603 |
| correct | 295 | 0.491 | 0.676 | 0.199 | 0.806 | 0.157 | 0.820 | 0.627 | 0.933 | 0.625 | 0.981 | 0.993 | 0.929 | 0.606 | 0.147 | 0.713 | 0.689 | 0.604 |
| misclassified | 5 | 0.482 | 0.712 | 0.002 | 0.000 | 0.000 | 0.246 | 0.447 | 0.640 | 0.452 | 0.961 | 0.986 | 0.760 | 0.625 | 0.172 | 0.642 | 0.775 | 0.752 |

| Unnamed: 0 | 0 |
|---|---|
| deletion_sal_lt_random_p | 1.000000 |
| insertion_sal_gt_random_p | 0.000000 |
| leaf_energy_gt_leaf_area_p | 0.000000 |

## External validation (s08)
| dataset | model | n | class_counts | accuracy | acc_ci_lo | acc_ci_hi | macro_precision | macro_recall | macro_f1 | recall_Early blight | recall_Late blight | recall_Healthy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PLD_test | MRV-Net | 405 | [162, 141, 102] | 0.6296 | 0.5816 | 0.6752 | 0.7003 | 0.6546 | 0.6270 | 0.3765 | 0.8227 | 0.7647 |
| PLD_test | VGG16 | 405 | [162, 141, 102] | 0.6000 | 0.5516 | 0.6466 | 0.6150 | 0.5767 | 0.5775 | 0.6296 | 0.7376 | 0.3627 |
| PLD_test | ResNet50 | 405 | [162, 141, 102] | 0.6148 | 0.5666 | 0.6609 | 0.7958 | 0.6242 | 0.6088 | 0.3210 | 0.9929 | 0.5588 |
| PLD_test | MobileNetV2 | 405 | [162, 141, 102] | 0.5778 | 0.5292 | 0.6249 | 0.6786 | 0.6105 | 0.5493 | 0.2284 | 0.8582 | 0.7451 |
| PLD_test | Xception | 405 | [162, 141, 102] | 0.6469 | 0.5992 | 0.6919 | 0.6937 | 0.6298 | 0.6267 | 0.5741 | 0.8936 | 0.4216 |
| PLD_test | DenseNet201 | 405 | [162, 141, 102] | 0.5778 | 0.5292 | 0.6249 | 0.7413 | 0.5670 | 0.5584 | 0.3951 | 0.9433 | 0.3627 |
| PLD_test | EfficientNetB0 | 405 | [162, 141, 102] | 0.5531 | 0.5044 | 0.6008 | 0.6800 | 0.5931 | 0.4943 | 0.0864 | 0.9574 | 0.7353 |
| PLD_test | NASNetLarge | 405 | [162, 141, 102] | 0.4346 | 0.3871 | 0.4832 | 0.5969 | 0.4443 | 0.3784 | 0.0864 | 0.8936 | 0.3529 |
| PLD_test | DenseNet121 | 405 | [162, 141, 102] | 0.5407 | 0.4921 | 0.5887 | 0.7789 | 0.5171 | 0.4879 | 0.3765 | 0.9787 | 0.1961 |
| PLD_all | MRV-Net | 4072 | [1628, 1424, 1020] | 0.6532 | 0.6385 | 0.6677 | 0.7114 | 0.6676 | 0.6530 | 0.4699 | 0.8202 | 0.7127 |
| PLD_all | VGG16 | 4072 | [1628, 1424, 1020] | 0.6194 | 0.6043 | 0.6341 | 0.6215 | 0.5867 | 0.5836 | 0.6953 | 0.7521 | 0.3127 |
| PLD_all | ResNet50 | 4072 | [1628, 1424, 1020] | 0.6098 | 0.5947 | 0.6246 | 0.7674 | 0.6117 | 0.6050 | 0.3729 | 0.9515 | 0.5108 |
| PLD_all | MobileNetV2 | 4072 | [1628, 1424, 1020] | 0.6017 | 0.5865 | 0.6166 | 0.6801 | 0.6224 | 0.5842 | 0.3286 | 0.8553 | 0.6833 |
| PLD_all | Xception | 4072 | [1628, 1424, 1020] | 0.6528 | 0.6380 | 0.6672 | 0.6913 | 0.6338 | 0.6316 | 0.5946 | 0.8841 | 0.4225 |
| PLD_all | DenseNet201 | 4072 | [1628, 1424, 1020] | 0.5833 | 0.5680 | 0.5983 | 0.7349 | 0.5677 | 0.5601 | 0.4251 | 0.9389 | 0.3392 |
| PLD_all | EfficientNetB0 | 4072 | [1628, 1424, 1020] | 0.5636 | 0.5483 | 0.5788 | 0.6616 | 0.5951 | 0.5188 | 0.1480 | 0.9508 | 0.6863 |
| PLD_all | NASNetLarge | 4072 | [1628, 1424, 1020] | 0.4649 | 0.4496 | 0.4802 | 0.6359 | 0.4707 | 0.4181 | 0.1333 | 0.9199 | 0.3588 |
| PLD_all | DenseNet121 | 4072 | [1628, 1424, 1020] | 0.5526 | 0.5372 | 0.5678 | 0.7837 | 0.5256 | 0.5001 | 0.4097 | 0.9719 | 0.1951 |

## De-duplicated test subset (s04)
_not available_

_not available_
