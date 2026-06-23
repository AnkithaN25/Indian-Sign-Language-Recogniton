"""
Transfer learning setup and training launcher for SSD MobileNetV2.

This module:
    1. Copies the pretrained pipeline.config to the custom model folder.
    2. Patches it for transfer learning (num_classes, checkpoint path, TF records).
    3. Prints the command to launch training (must be run in a terminal).

Usage
-----
    python3 src/train.py
    # Then copy-paste the printed training command into your terminal.
"""

import os
import shutil

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util

from src.config import (
    ANNOTATION_PATH,
    APIMODEL_PATH,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    CUSTOM_MODEL_NAME,
    LABEL_MAP_PATH,
    MODEL_PATH,
    NUM_CLASSES,
    NUM_TRAIN_STEPS,
    PRETRAINED_CHECKPOINT,
    PRETRAINED_CONFIG,
    TEST_RECORD,
    TRAIN_RECORD,
)


def copy_pretrained_config() -> None:
    """Copy the pretrained model's pipeline.config to the custom model directory."""
    dest_dir = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(PRETRAINED_CONFIG, dest_dir)
    print(f"Copied pretrained config → {dest_dir}")


def update_pipeline_config() -> None:
    """
    Patch pipeline.config for transfer learning:
        - num_classes         → number of ISL gesture labels
        - batch_size          → from config.BATCH_SIZE
        - fine_tune_checkpoint → pretrained checkpoint path
        - label_map_path      → our label_map.pbtxt
        - train/eval TF records → our .record files
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:
        text_format.Merge(f.read(), pipeline_config)

    # Patch fields
    pipeline_config.model.ssd.num_classes = NUM_CLASSES
    pipeline_config.train_config.batch_size = BATCH_SIZE
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_CHECKPOINT
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path = LABEL_MAP_PATH
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [TRAIN_RECORD]
    pipeline_config.eval_input_reader[0].label_map_path = LABEL_MAP_PATH
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [TEST_RECORD]

    config_text = text_format.MessageToString(pipeline_config)
    with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:
        f.write(config_text)
    print(f"Pipeline config updated → {CONFIG_PATH}")


def print_training_command() -> None:
    """Print the command to start model training."""
    cmd = (
        f"python {APIMODEL_PATH}/research/object_detection/model_main_tf2.py "
        f"--model_dir={MODEL_PATH}/{CUSTOM_MODEL_NAME} "
        f"--pipeline_config_path={CONFIG_PATH} "
        f"--num_train_steps={NUM_TRAIN_STEPS}"
    )
    print("\nRun the following command to start training:\n")
    print(cmd)
    print()


if __name__ == "__main__":
    copy_pretrained_config()
    update_pipeline_config()
    print_training_command()
