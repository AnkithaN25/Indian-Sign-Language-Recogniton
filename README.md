# Indian Sign Language Detection

Real-time Indian Sign Language (ISL) gesture recognition using **SSD MobileNetV2** transfer learning via the TensorFlow Object Detection API. Detects hand gestures from a live webcam feed and labels them with bounding boxes.

---

## Model

**Architecture:** SSD MobileNetV2 FPN-Lite 320×320 (pretrained on COCO 2017)  
**Fine-tuned on:** Custom ISL gesture dataset (3 gesture classes: 1, 2, 3)  
**Training:** 5,000 steps, batch size 4, transfer learning from `ckpt-0`  
**Inference:** Real-time webcam, ~30 FPS on CPU

---

## Project Structure

```
indian-sign-language-detection/
│
├── Tensorflow/
│   ├── scripts/
│   │   └── generate_tfrecord.py     # TF Record generator (from OD API)
│   └── workspace/
│       ├── annotations/             # label_map.pbtxt + .record files
│       ├── images/
│       │   ├── train/               # Annotated training images (.jpg + .xml)
│       │   └── test/                # Annotated test images (.jpg + .xml)
│       ├── models/
│       │   └── my_ssd_mobnet/       # Trained checkpoints + pipeline.config
│       └── pre-trained-models/      # Downloaded pretrained SSD MobileNetV2
│
├── notebooks/
│   └── indian_sign_language_detection.ipynb   # Interactive walkthrough
│
├── src/
│   ├── config.py    # All paths, labels, and hyperparameters
│   ├── setup.py     # create_label_map(), generate_tf_records()
│   ├── train.py     # update_pipeline_config(), print_training_command()
│   └── detect.py    # load_model(), run_realtime_detection()
│
├── requirements.txt
├── .gitignore
└── README.md
```

### `src/` vs notebook

All logic lives in `src/`. The notebook calls `src/` functions with explanations. You can run the full pipeline from the command line without opening Jupyter.

---

## How to Run

### Step 1 — Clone this repo and install dependencies

```bash
git clone https://github.com/your-username/indian-sign-language-detection.git
cd indian-sign-language-detection
pip3 install -r requirements.txt
```

### Step 2 — Install the TF Object Detection API

```bash
cd Tensorflow
git clone https://github.com/tensorflow/models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
python setup.py install
pip install .
cd ../../..
```

### Step 3 — Download pretrained SSD MobileNetV2

```bash
cd Tensorflow/workspace/pre-trained-models
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
tar -xvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
cd ../../..
```

### Step 4 — Add your training images

Place annotated images in:
```
Tensorflow/workspace/images/train/    # .jpg + .xml (Pascal VOC format)
Tensorflow/workspace/images/test/     # .jpg + .xml
```

Use [LabelImg](https://github.com/HumanSignal/labelImg) to annotate. Save annotations in **Pascal VOC XML** format.

Update `src/config.py` to match your gesture labels:
```python
LABELS = [
    {"name": "gesture_a", "id": 1},
    {"name": "gesture_b", "id": 2},
    {"name": "gesture_c", "id": 3},
]
```

### Step 5 — Create label map and TF Records

```bash
python3 src/setup.py
```

### Step 6 — Configure and start training

```bash
python3 src/train.py
# Copy the printed command and run it in your terminal
```

Training progress can be monitored with TensorBoard:
```bash
tensorboard --logdir=Tensorflow/workspace/models/my_ssd_mobnet
```

### Step 7 — Run real-time detection

```bash
python3 src/detect.py
# Press 'q' to quit

# Optional flags:
python3 src/detect.py --checkpoint ckpt-6 --threshold 0.5
```

---

## Configuration

All paths, labels, and hyperparameters are in `src/config.py`. Key settings:

| Setting | Default | Description |
|---|---|---|
| `LABELS` | 3 gesture classes | Add/rename to match your ISL gestures |
| `NUM_TRAIN_STEPS` | 5000 | Increase for better accuracy |
| `BATCH_SIZE` | 4 | Reduce if running out of memory |
| `INFERENCE_CHECKPOINT` | `ckpt-6` | Which checkpoint to use for detection |
| `MIN_SCORE_THRESH` | 0.5 | Minimum confidence to show a detection |

---

## Extending to More Gestures

1. Add new entries to `LABELS` in `src/config.py`
2. Collect and annotate images for each new gesture
3. Re-run `python3 src/setup.py` to regenerate the label map and TF Records
4. Re-run training with `python3 src/train.py`

---

## Dependencies

- TensorFlow 2.9+
- OpenCV
- TF Object Detection API (from `tensorflow/models`)
- Pretrained SSD MobileNetV2 FPN-Lite 320×320 (COCO 2017)

---

## References

- [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [SSD MobileNetV2 FPN-Lite — TF Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- [LabelImg — Image Annotation Tool](https://github.com/HumanSignal/labelImg)
