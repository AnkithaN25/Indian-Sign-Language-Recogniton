"""
Setup utilities: create the label map and generate TF Records.

Run this once before training whenever you add new labels or images.

Usage
-----
    python3 src/setup.py
"""

import os
import subprocess
import sys

from src.config import (
    ANNOTATION_PATH,
    IMAGE_PATH,
    LABEL_MAP_PATH,
    LABELS,
    SCRIPTS_PATH,
    TEST_RECORD,
    TRAIN_RECORD,
)


def create_label_map(labels: list[dict], output_path: str = LABEL_MAP_PATH) -> None:
    """
    Write a TensorFlow Object Detection label map (.pbtxt) file.

    Args:
        labels:      List of dicts with 'name' and 'id' keys (from config.LABELS).
        output_path: Where to write the .pbtxt file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for label in labels:
            f.write("item {\n")
            f.write(f"\tname:'{label['name']}'\n")
            f.write(f"\tid:{label['id']}\n")
            f.write("}\n")
    print(f"Label map written → {output_path}")


def generate_tf_records(
    scripts_path: str = SCRIPTS_PATH,
    image_path: str = IMAGE_PATH,
    label_map_path: str = LABEL_MAP_PATH,
    train_record: str = TRAIN_RECORD,
    test_record: str = TEST_RECORD,
) -> None:
    """
    Run generate_tfrecord.py for both train and test splits.

    Requires:
        - Annotated images in Tensorflow/workspace/images/train/ and /test/
        - Tensorflow/scripts/generate_tfrecord.py (from the TF Object Detection repo)
        - label_map.pbtxt already created (call create_label_map() first)
    """
    generate_script = os.path.join(scripts_path, "generate_tfrecord.py")

    for split, record_path in [("train", train_record), ("test", test_record)]:
        img_dir = os.path.join(image_path, split)
        cmd = [
            sys.executable, generate_script,
            "-x", img_dir,
            "-l", label_map_path,
            "-o", record_path,
        ]
        print(f"Generating {split} TF Record…")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating {split} record:\n{result.stderr}")
        else:
            print(f"  {split} record → {record_path}")


if __name__ == "__main__":
    from src.config import LABELS
    create_label_map(LABELS)
    generate_tf_records()
