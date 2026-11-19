"""
Configuration for the MRV-Net experiments.
Edit the paths in this file only; every script reads from here.
"""
import os

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
# The published Train/Valid/Test split is used. Two ways to provide it:
#   (a) SPLIT_SOURCE = "folders": DATA_ROOT/{train,val,test}/<class folder>/*.jpg
#   (b) SPLIT_SOURCE = "csv":     an existing CSV with columns path,label,split
#       (split in {train,val,test}; label in 0,1,2), e.g. exported from another pipeline.
SPLIT_SOURCE = "folders"
# Kaggle dataset used in the paper (Putra, 2020):
# https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset
# It is distributed already split (900 / 300 / 300) inside the "Potato" folder.
# Point DATA_ROOT at that folder. Split and class folder names are matched
# case-insensitively (Train/Training, Valid/Val/Validation, Test/Testing).
DATA_ROOT = os.path.join("data", "Potato")   # folder containing Train, Valid, Test (Kaggle: /kaggle/input/potato-leaf-disease-dataset/Potato)
SPLIT_FOLDER_NAMES = {"train": "Train", "val": "Valid", "test": "Test"}
EXISTING_SPLIT_CSV = "my_split.csv"   # used when SPLIT_SOURCE == "csv"

# Class folder names in label order 0,1,2 (Table 1 of the manuscript)
CLASS_DIRS = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]  # matched case-insensitively
CLASS_NAMES = ["Early blight", "Late blight", "Healthy"]
IMG_SIZE = 224           # MODEL input size used in training (images on disk are 256x256 and are resized)

# ---------------------------------------------------------------------------
# 2. SAVED MODELS (written by s00b_train_models.py)
# ---------------------------------------------------------------------------
import keras as _k
MODEL_EXT = ".keras" if int(_k.__version__.split(".")[0]) >= 3 else ".h5"
WEIGHTS = "imagenet"     # pretrained weights for every backbone
MRVNET_PATH = "models/mrvnet" + MODEL_EXT

# The eight individually trained transfer-learning baselines.
# Missing files are skipped automatically (comparisons are then limited to
# the models that are available).
BASELINE_PATHS = {
    "VGG16": "models/vgg16" + MODEL_EXT,
    "ResNet50": "models/resnet50" + MODEL_EXT,
    "MobileNetV2": "models/mobilenetv2" + MODEL_EXT,
    "Xception": "models/xception" + MODEL_EXT,
    "DenseNet201": "models/densenet201" + MODEL_EXT,
    "EfficientNetB0": "models/efficientnetb0" + MODEL_EXT,
    "NASNetLarge": "models/nasnetlarge" + MODEL_EXT,
    "DenseNet121": "models/densenet121" + MODEL_EXT,
}

# How each saved model expects its input. The scripts load images as
# float32 RGB in [0, 255] at IMG_SIZE x IMG_SIZE and then apply this mode:
#   "raw"   -> pass 0-255 pixels unchanged (preprocessing is inside the model)
#   "unit"  -> divide by 255
#   "<name>"-> apply keras.applications.<name>.preprocess_input
#              e.g. "resnet50", "vgg16", "mobilenet_v2", "xception",
#              "densenet", "efficientnet", "nasnet"
# Run s01_model_summary.py first: it prints the layers so you can confirm
# whether the preprocessing Lambda layers are inside MRV-Net.
MRVNET_INPUT_MODE = "raw"   # models trained by s00b contain their own preprocessing
BASELINE_INPUT_MODE = {k: "raw" for k in BASELINE_PATHS}  # preprocessing is inside every model trained by s00b
# NASNetLarge was built for 331x331; set a different size here if you trained it so.
BASELINE_IMG_SIZE = {}  # e.g. {"NASNetLarge": 331}

# Order of the branches inside MRV-Net's Concatenate layer (channel order).
# s01 prints the detected order; change only if it differs.
BRANCH_ORDER = ["MobileNetV2", "ResNet50", "VGG16"]
BRANCH_CHANNELS = {"MobileNetV2": 1280, "ResNet50": 2048, "VGG16": 512}
BRANCH_PREPROCESS = {"MobileNetV2": "mobilenet_v2", "ResNet50": "resnet50", "VGG16": "vgg16"}

# ---------------------------------------------------------------------------
# 3. HEAD-ONLY ABLATION PROTOCOL (backbones are frozen, so features are cached)
# ---------------------------------------------------------------------------
FEATURE_DIR = "cache_features"
FEATURE_BATCH = 16       # images per forward pass when caching features (lower if GPU memory is short)
AUG_COPIES = 5           # augmented copies cached per training image (plus the original)
SEEDS = [0, 1, 2, 3, 4]  # 5 independent runs; use 10 seeds if time allows
MAX_EPOCHS = 100
PATIENCE = 15            # early stopping on validation loss (validation set only)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT = 0.3
FUSION_ACTIVATION = "relu"   # activation of the 2x2 fusion Conv2D
FUSION_PADDING = "same"      # as implemented: 2x2 conv, stride 1, same padding (7x7 -> 7x7)
DETERMINISTIC_OPS = False    # True = bit-reproducible GPU ops (slower)

# Augmentation (Keras ImageDataGenerator semantics: rotation and shear in degrees)
AUGMENT = dict(rotation=10, shear=0.2, zoom=0.2, shift=0.2, hflip=True, vflip=False)

# ---------------------------------------------------------------------------
# 4. EXTERNAL VALIDATION (field-acquired images, never used for training)
# ---------------------------------------------------------------------------
# Example: Potato Leaf Disease (PLD) dataset, Rashid et al., Electronics 2021.
# Point to a folder with one sub-folder per class.
EXTERNAL_SETS = {
    # PLD (Rashid et al., Electronics 2021): field images, Central Punjab, Pakistan. Never used for training.
    "PLD_test": {   # the official PLD test split (405 images)
        "root": os.path.join("data", "PLD", "Test"),
        "class_dirs": ["Early_Blight", "Late_Blight", "Healthy"],  # label order 0,1,2
    },
    "PLD_all": {    # all PLD images (Train + Valid + Test), all unseen by our models
        "root": [os.path.join("data", "PLD", "Train"),
                 os.path.join("data", "PLD", "Valid"),
                 os.path.join("data", "PLD", "Test")],
        "class_dirs": ["Early_Blight", "Late_Blight", "Healthy"],
    },
}

# ---------------------------------------------------------------------------
# 5. XAI
# ---------------------------------------------------------------------------
EXPERT_MASK_DIR = None   # optional folder of binary lesion masks named like the test images (.png)
SHAP_MAX_EVALS = 300     # evaluations per image for the SHAP Partition explainer
SHAP_BATCH = 16         # forward-pass batch for SHAP (lower if GPU memory is short)
SHAP_ON_ALL_TEST = True  # False = only the 8 misclassified + 30 random correct images
TOP_FRACTION = 0.20      # top-k% salient pixels used for IoU with lesion masks

# ---------------------------------------------------------------------------
# 6. OUTPUT
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
SPLIT_CSV = os.path.join(RESULTS_DIR, "split.csv")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
RANDOM_STATE = 2026
