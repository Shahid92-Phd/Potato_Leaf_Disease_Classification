# Data

The images are not redistributed here. Download them from the original sources and
place them as follows (folder and class names are matched case-insensitively).

```
data/
  Potato/                      # Putra (2020), Kaggle: muhammadardiputra/potato-leaf-disease-dataset
    Train/ Potato___Early_blight/ Potato___Late_blight/ Potato___healthy/    (300 per class)
    Valid/ ...                                                               (100 per class)
    Test/  ...                                                               (100 per class)
  PLD/                         # Rashid et al. (2021), Electronics 10(17):2064 - external test only
    Train/ Early_Blight/ Late_Blight/ Healthy/
    Valid/ ...
    Test/  ...
```

* Kaggle (training/validation/internal test):
  https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset
  Use the published Train/Valid/Test split unchanged (900/300/300 images).
* PLD (external validation, never used for training or model selection): the
  Potato Leaf Disease dataset of Rashid et al. (2021), field images from Central
  Punjab, Pakistan. All 4,072 images (and, separately, its 405-image test split)
  are used only for inference.

`results/split.csv` lists every Kaggle image with its split and label, and
`results/provenance_check.md` documents the source-leaf audit of the healthy class.
