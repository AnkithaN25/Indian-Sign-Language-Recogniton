"""
Real-time Indian Sign Language detection via webcam.

Loads the fine-tuned SSD MobileNetV2 checkpoint and runs inference
on each webcam frame, displaying bounding boxes and gesture labels.

Usage
-----
    python3 src/detect.py
    python3 src/detect.py --checkpoint ckpt-6 --threshold 0.5
    # Press 'q' to quit.
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utils

from src.config import (
    ANNOTATION_PATH,
    CHECKPOINT_PATH,
    CONFIG_PATH,
    CUSTOM_MODEL_NAME,
    DISPLAY_SIZE,
    INFERENCE_CHECKPOINT,
    LABEL_MAP_PATH,
    MAX_BOXES,
    MIN_SCORE_THRESH,
    MODEL_PATH,
)


def load_model(checkpoint: str = INFERENCE_CHECKPOINT):
    """
    Build the SSD detection model and restore weights from checkpoint.

    Args:
        checkpoint: Checkpoint name to restore, e.g. 'ckpt-6'.

    Returns:
        Loaded detection model.
    """
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(
        model_config=configs["model"], is_training=False
    )

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, checkpoint)).expect_partial()
    print(f"Model loaded from {CHECKPOINT_PATH}/{checkpoint}")
    return detection_model


@tf.function
def detect_fn(detection_model, image: tf.Tensor) -> dict:
    """
    Run a single forward pass through the detection model.

    Args:
        detection_model: Loaded TF object detection model.
        image:           4-D float tensor, shape (1, H, W, 3).

    Returns:
        Dict of detection results (boxes, classes, scores, num_detections).
    """
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    return detection_model.postprocess(prediction_dict, shapes)


def run_realtime_detection(
    checkpoint: str = INFERENCE_CHECKPOINT,
    threshold: float = MIN_SCORE_THRESH,
) -> None:
    """
    Open webcam and run real-time ISL gesture detection.

    Args:
        checkpoint: Checkpoint name to load (e.g. 'ckpt-6').
        threshold:  Minimum confidence score to display a detection box.

    Controls:
        Press 'q' to quit.
    """
    detection_model  = load_model(checkpoint)
    category_index   = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that a camera is connected.")

    print("Starting real-time detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame — exiting.")
            break

        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32
        )

        detections = detect_fn(detection_model, input_tensor)

        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections["detection_boxes"],
            detections["detection_classes"] + 1,   # label_id_offset = 1
            detections["detection_scores"],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=MAX_BOXES,
            min_score_thresh=threshold,
            agnostic_mode=False,
        )

        cv2.imshow(
            "Indian Sign Language Detection",
            cv2.resize(image_np_with_detections, DISPLAY_SIZE),
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time ISL gesture detection.")
    parser.add_argument(
        "--checkpoint", default=INFERENCE_CHECKPOINT,
        help="Checkpoint to load for inference (default: %(default)s)"
    )
    parser.add_argument(
        "--threshold", type=float, default=MIN_SCORE_THRESH,
        help="Minimum confidence score to display a detection (default: %(default)s)"
    )
    args = parser.parse_args()
    run_realtime_detection(checkpoint=args.checkpoint, threshold=args.threshold)
