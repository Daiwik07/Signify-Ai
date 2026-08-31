import os
import sys
from contextlib import contextmanager
from pathlib import Path
import urllib.request

# Reduce TensorFlow Lite / MediaPipe native logging.
# These must be set before importing MediaPipe.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")

import cv2
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


# ---------------------------------------------------------------------------
# Signify AI visual polish
# ---------------------------------------------------------------------------
# app.py and collect_data.py both use the same imported cv2 module. By adding
# the small wrappers below here, we can make only the Signify AI app windows
# full-screen and make its UI wording friendlier without changing any model or
# training behavior.
_ORIGINAL_IMSHOW = cv2.imshow
_ORIGINAL_PUT_TEXT = cv2.putText
_ORIGINAL_DESTROY_ALL_WINDOWS = cv2.destroyAllWindows
_FULLSCREEN_WINDOWS = set()


def _humanize_ui_text(text):
    """Turn technical UI labels into simple, natural teaching language."""
    text = str(text)

    exact_replacements = {
        "PHOTO GUIDE": "How to Make This Sign",
        "Copy the photo": "Try to copy this hand shape",
        "REFERENCE": "Practice Guide",
        "Copy this hand shape": "Match your hand to this guide",
        "Movement detected": "Nice - I can see the movement",
        "Hand mostly still": "Hold your hand naturally",
        "CORRECT! +1": "Correct! Well done!",
        "CORRECT! Great job!": "That's right! Great job!",
        "SIGNIFY AI - IDENTIFIER": "Signify AI - Live Sign Reader",
        "SPACE=add   C=clear   Q=return": "SPACE: add sign   C: clear   Q: back",
    }

    if text in exact_replacements:
        return exact_replacements[text]

    if text.startswith("Learn: "):
        return "Let's learn: " + text[len("Learn: "):]

    if text.startswith("AI sees: "):
        return "I can see: " + text[len("AI sees: "):]

    if text.startswith("Pose similarity: "):
        value = text[len("Pose similarity: "):]
        try:
            percent = float(value.rstrip("%"))
        except ValueError:
            return "Your match: " + value

        if percent >= 90:
            return f"Excellent match: {percent:.0f}%"
        if percent >= 75:
            return f"Great, almost there: {percent:.0f}%"
        if percent >= 55:
            return f"Good try, adjust a little: {percent:.0f}%"
        return f"Keep going, your match is {percent:.0f}%"

    if text.startswith("MAKE: "):
        return "Show me: " + text[len("MAKE: "):]

    if text.startswith("Time: "):
        return "Time left: " + text[len("Time: "):]

    # Identifier uses "Sign: X". collect_data.py uses
    # "Sign: X   Samples: ...", so leave the data-collection screen unchanged.
    if text.startswith("Sign: ") and "Samples:" not in text:
        return "I can see: " + text[len("Sign: "):]

    if text.startswith("Confidence: "):
        return "I'm " + text[len("Confidence: "):] + " sure"

    if text.startswith("Sequence: "):
        return "Your signs: " + text[len("Sequence: "):]

    return text


def _signify_put_text(image, text, *args, **kwargs):
    return _ORIGINAL_PUT_TEXT(
        image,
        _humanize_ui_text(text),
        *args,
        **kwargs,
    )


def _signify_imshow(window_name, image):
    """Open Signify AI webcam screens in full-screen mode."""
    name = str(window_name)

    if name.startswith("Signify AI") and name not in _FULLSCREEN_WINDOWS:
        try:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
            _FULLSCREEN_WINDOWS.add(name)
        except cv2.error:
            # If a platform does not support OpenCV full-screen, keep the app
            # usable in a normal window rather than crashing.
            pass

    return _ORIGINAL_IMSHOW(window_name, image)


def _signify_destroy_all_windows():
    _FULLSCREEN_WINDOWS.clear()
    return _ORIGINAL_DESTROY_ALL_WINDOWS()


# Apply the UI helpers to the shared OpenCV module.
cv2.putText = _signify_put_text
cv2.imshow = _signify_imshow
cv2.destroyAllWindows = _signify_destroy_all_windows


@contextmanager
def suppress_native_stderr():
    """
    Temporarily silence native C/C++ messages written directly to stderr.

    MediaPipe/TensorFlow Lite can print INFO/WARNING messages from native code
    even when Python logging is disabled. This redirects only stderr while a
    MediaPipe native call is running. If a real Python exception occurs, the
    traceback is still shown after stderr is restored.
    """
    try:
        stderr_fd = sys.stderr.fileno()
        sys.stderr.flush()
        saved_fd = os.dup(stderr_fd)
        null_fd = os.open(os.devnull, os.O_WRONLY)
    except (AttributeError, OSError, ValueError):
        # Some IDE consoles do not expose a normal stderr file descriptor.
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
    """Small wrapper that keeps MediaPipe's native console noise hidden."""

    def __init__(self, landmarker):
        self._landmarker = landmarker

    def detect_for_video(self, image, timestamp_ms):
        with suppress_native_stderr():
            return self._landmarker.detect_for_video(image, timestamp_ms)

    def close(self):
        with suppress_native_stderr():
            return self._landmarker.close()

    def __getattr__(self, name):
        return getattr(self._landmarker, name)


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

    # Model creation itself can print XNNPACK / feedback-manager messages.
    with suppress_native_stderr():
        landmarker = HandLandmarker.create_from_options(options)

    return QuietHandLandmarker(landmarker)


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
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

    return frame
