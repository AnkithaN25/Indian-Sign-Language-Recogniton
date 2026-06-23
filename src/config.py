"""
Central configuration: all paths, label definitions, and training hyperparameters.
Every other module imports from here — change values once, they propagate everywhere.
"""

import os

# ── Workspace layout ──────────────────────────────────────────────────────────
WORKSPACE_PATH        = "Tensorflow/workspace"
SCRIPTS_PATH          = "Tensorflow/scripts"
APIMODEL_PATH         = "Tensorflow/models"
ANNOTATION_PATH       = os.path.join(WORKSPACE_PATH, "annotations")
IMAGE_PATH            = os.path.join(WORKSPACE_PATH, "images")
MODEL_PATH            = os.path.join(WORKSPACE_PATH, "models")
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH, "pre-trained-models")

# ── Model identifiers ─────────────────────────────────────────────────────────
CUSTOM_MODEL_NAME     = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"

# ── Derived paths ─────────────────────────────────────────────────────────────
CONFIG_PATH     = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, "pipeline.config")
CHECKPOINT_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)
PRETRAINED_CHECKPOINT = os.path.join(
    PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME, "checkpoint", "ckpt-0"
)
PRETRAINED_CONFIG = os.path.join(
    PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME, "pipeline.config"
)

LABEL_MAP_PATH  = os.path.join(ANNOTATION_PATH, "label_map.pbtxt")
TRAIN_RECORD    = os.path.join(ANNOTATION_PATH, "train.record")
TEST_RECORD     = os.path.join(ANNOTATION_PATH, "test.record")

# ── Label definitions ─────────────────────────────────────────────────────────
# Update names and add/remove entries to match your ISL gestures.
# 'name' is the gesture label shown during inference.
# 'id'   must be a consecutive integer starting from 1.
LABELS = [
    {"name": "1", "id": 1},
    {"name": "2", "id": 2},
    {"name": "3", "id": 3},
]
NUM_CLASSES = len(LABELS)

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE       = 4
NUM_TRAIN_STEPS  = 5000
CHECKPOINT_EVERY = 1000    # save a checkpoint every N steps

# ── Inference settings ────────────────────────────────────────────────────────
INFERENCE_CHECKPOINT = "ckpt-6"     # which checkpoint to load for inference
MIN_SCORE_THRESH     = 0.5          # minimum confidence to display a detection
MAX_BOXES            = 5
DISPLAY_SIZE         = (800, 600)   # window size for real-time detection
