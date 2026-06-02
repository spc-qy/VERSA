# ======================================================================
# 0. CPU + OpenBLAS safety settings
# This section must be placed first.
# ======================================================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import torch
torch.backends.mkldnn.enabled = False

import cv2
cv2.setNumThreads(0)


# ======================================================================
# 1. Project path configuration
# ======================================================================
# Project root directory, for example: sheep-vertebra/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directory
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    str(PROJECT_ROOT / "data")
)

# Heatmap prior directory
PRIORS_ROOT = os.environ.get(
    "PRIORS_ROOT",
    str(Path(DATA_ROOT) / "_priors")
)

# Output directory
# By default, results are saved to the outputs/ directory under the project root.
SAVE_DIR = os.environ.get(
    "SAVE_DIR",
    str(PROJECT_ROOT / "outputs")
)

os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================================================
# 2. Basic training configuration
# ======================================================================
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
DEVICE = "cpu"

# Outer 5-fold cross-validation
N_SPLITS = 5
CV_RANDOM_STATE = 42

# Split a small validation set from the outer training set
INNER_VAL_RATIO = 0.125

# Two-stage fine-tuning parameters
STAGE1_EPOCHS = 8
LR_STAGE1 = 1e-4
LR_STAGE2 = 3e-5

# Top-k checkpoint averaging
TOP_K = 3


# ======================================================================
# 3. Heatmap structural feature branch parameters
# ======================================================================
MAX_POINTS = 25
HM_PEAK_REL_THRESH = 0.30
HM_PEAK_MIN_DIST = 5


# ======================================================================
# 4. Fusion weights for image features and structural features
# ======================================================================
FUSION_ALPHA = float(os.environ.get("FUSION_ALPHA", "1.0"))

IMAGE_FEAT_WEIGHT = FUSION_ALPHA
STRUCT_FEAT_WEIGHT = 1.0 - FUSION_ALPHA

# Determine whether to enable the structural branch
USE_STRUCT_BRANCH = IMAGE_FEAT_WEIGHT < 1.0

USE_PRIOR_FOR_VAL_TEST = False
