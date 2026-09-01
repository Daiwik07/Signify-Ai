import os
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")

import cv2
import mediapipe as mp
import numpy as np

MODEL_DIR = Path("models")
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


@contextmanager
def quiet_native_logs():
    """Hide the noisy MediaPipe/TFLite messages printed by native code."""
    try:
        stderr_fd = sys.stderr.fileno()
        sys.stderr.flush()
        saved_fd = os.dup(stderr_fd)
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except (AttributeError, OSError, ValueError):
        yield
        return

    try:
        os.dup2(null_fd, stderr_fd)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(null_fd)


class QuietHandLandmarker:
    def __init__(self, landmarker):
        self.landmarker = landmarker

    def detect_for_video(self, image, timestamp_ms):
        with quiet_native_logs():
            return self.landmarker.detect_for_video(image, timestamp_ms)

    def close(self):
        with quiet_native_logs():
            self.landmarker.close()


def ensure_hand_model():
    MODEL_DIR.mkdir(exist_ok=True)

    if not HAND_MODEL.exists():
        print("Downloading MediaPipe hand model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
        print("Downloaded:", HAND_MODEL)

    return HAND_MODEL


def create_landmarker():
    model_path = ensure_hand_model()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with quiet_native_logs():
        landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    return QuietHandLandmarker(landmarker)


def extract_features(hand_landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )

    points -= points[0]

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0

    return (points / scale).flatten()


def draw_hand(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(lm.x * width), int(lm.y * height))
        for lm in hand_landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)

    return frame