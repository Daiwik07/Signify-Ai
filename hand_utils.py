from pathlib import Path
import urllib.request
import numpy as np
import mediapipe as mp

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

def ensure_hand_model():
    MODEL_DIR.mkdir(exist_ok=True)
    if not HAND_MODEL.exists():
        print("Downloading MediaPipe hand model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
        print("Downloaded:", HAND_MODEL)
    return HAND_MODEL

def create_landmarker():
    model_path = ensure_hand_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)

def extract_features(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks into 63 normalized features.
    We make the wrist the origin and normalize by the largest distance.
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32,
    )
    points = points - points[0]  # wrist becomes (0, 0, 0)

    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale < 1e-6:
        scale = 1.0

    points = points / scale
    return points.flatten()

def draw_hand(frame, hand_landmarks):
    import cv2

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    return frame
