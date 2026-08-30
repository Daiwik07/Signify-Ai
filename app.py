import time
from collections import deque, Counter
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from hand_utils import create_landmarker, extract_features, draw_hand

MODEL_FILE = Path("models/sign_classifier.joblib")
WAVE_MOTION_THRESHOLD = 0.015


def smooth_prediction(history):
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]


def wrist_motion_score(wrist_history):
    if len(wrist_history) < 2:
        return 0.0

    points = np.array(wrist_history, dtype=np.float32)
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.mean(deltas))

def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python train_model.py"
        )

    model = joblib.load(MODEL_FILE)
    landmarker = create_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    history = deque(maxlen=8)
    wrist_history = deque(maxlen=6)

    print("AI Sign Language Helper")
    print("Press Q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            label = "No hand"
            confidence = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)
                wrist_history.append((hand[0].x, hand[0].y))

                motion = wrist_motion_score(wrist_history)

                features = extract_features(hand).reshape(1, -1)
                probabilities = model.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))
                raw_label = model.classes_[best_index]
                confidence = float(probabilities[best_index])

                if confidence >= 0.55:
                    history.append(raw_label)
                    label = smooth_prediction(history) or raw_label
                else:
                    history.clear()
                    label = "Not sure"
            else:
                history.clear()
                wrist_history.clear()

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"AI Sign: {label}",
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            if label not in ("No hand", "Not sure"):
                cv2.putText(
                    frame,
                    f"Confidence: {confidence * 100:.1f}%",
                    (18, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Show a trained sign",
                    (18, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("AI Sign Language Helper", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
