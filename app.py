import csv
import random
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from hand_utils import HAND_CONNECTIONS, create_landmarker, draw_hand, extract_features

MODEL_FILE = Path("models/sign_classifier.joblib")
DATA_FILE = Path("data/signs.csv")

CONFIDENCE_THRESHOLD = 0.60
HOLD_FRAMES = 6
WAVE_MOTION_THRESHOLD = 0.015


def normalize_label(text):
    return "_".join(text.strip().upper().replace("-", " ").split())


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


def load_model():
    if not MODEL_FILE.exists():
        print("\nTrained model not found.")
        print("Collect at least two signs and then run: python train_model.py")
        return None
    return joblib.load(MODEL_FILE)


def trained_labels(model):
    return {normalize_label(str(x)): str(x) for x in model.classes_}


def predict_hand(model, hand):
    features = extract_features(hand)
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    best = int(np.argmax(probabilities))
    return str(model.classes_[best]), float(probabilities[best]), features


def load_reference_pose(label):
    """Average your saved training samples to create a teaching pose."""
    if not DATA_FILE.exists():
        return None

    wanted = normalize_label(label)
    rows = []

    with DATA_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = [f"f{i}" for i in range(63)]
        for row in reader:
            if normalize_label(row.get("label", "")) != wanted:
                continue
            try:
                rows.append([float(row[name]) for name in fields])
            except (KeyError, TypeError, ValueError):
                pass

    if not rows:
        return None

    return np.mean(np.asarray(rows, dtype=np.float32), axis=0).reshape(21, 3)


def reference_panel(label, pose, height):
    width = 330
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.putText(panel, "REFERENCE", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(panel, label.replace("_", " "), (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if pose is None:
        cv2.putText(panel, "No reference pose", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return panel

    points = pose[:, :2]
    extent = max(float(np.max(np.abs(points))), 1e-6)
    usable_h = max(height - 160, 120)
    scale = min(width - 70, usable_h) * 0.48 / extent
    cx, cy = width // 2, 110 + usable_h // 2

    pts = [(int(cx + x * scale), int(cy + y * scale)) for x, y in points]

    for a, b in HAND_CONNECTIONS:
        cv2.line(panel, pts[a], pts[b], (255, 255, 255), 3)
    for x, y in pts:
        cv2.circle(panel, (x, y), 6, (0, 255, 0), -1)

    cv2.putText(panel, "Copy this hand shape", (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return panel


def pose_similarity(features, reference):
    if reference is None:
        return None
    current = np.asarray(features, dtype=np.float32).reshape(21, 3)
    distance = float(np.mean(
        np.linalg.norm(current[:, :2] - reference[:, :2], axis=1)
    ))
    return max(0.0, min(100.0, 100.0 * (1.0 - distance / 0.45)))


def practice_sign(model, target):
    target = normalize_label(target)
    if target not in trained_labels(model):
        return False

    reference = load_reference_pose(target)
    landmarker = create_landmarker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        landmarker.close()
        print("Could not open webcam.")
        return False

    history = deque(maxlen=8)
    wrist_history = deque(maxlen=6)
    correct_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(
                image, int(time.monotonic() * 1000)
            )

            label = "No hand"
            confidence = 0.0
            similarity = None
            motion = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)

                wrist_history.append((hand[0].x, hand[0].y))
                motion = wrist_motion_score(wrist_history)

                raw, confidence, features = predict_hand(model, hand)
                similarity = pose_similarity(features, reference)

                if confidence >= CONFIDENCE_THRESHOLD:
                    history.append(raw)
                    label = smooth_prediction(history) or raw
                else:
                    history.clear()
                    label = "Not sure"

                if normalize_label(label) == target and confidence >= CONFIDENCE_THRESHOLD:
                    correct_frames += 1
                else:
                    correct_frames = 0
            else:
                history.clear()
                wrist_history.clear()
                correct_frames = 0

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 135), (0, 0, 0), -1)
            cv2.putText(frame, f"Learn: {target.replace('_', ' ')}", (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
            cv2.putText(frame, f"AI sees: {label}", (15, 67),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)

            if similarity is not None:
                cv2.putText(frame, f"Pose similarity: {similarity:.0f}%", (15, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

            motion_text = "Movement detected" if motion >= WAVE_MOTION_THRESHOLD else "Hand mostly still"
            cv2.putText(frame, motion_text, (15, 126),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

            combined = np.hstack([
                frame,
                reference_panel(target, reference, frame.shape[0])
            ])

            if correct_frames >= HOLD_FRAMES:
                cv2.rectangle(combined, (0, combined.shape[0] - 70),
                              (combined.shape[1], combined.shape[0]),
                              (0, 0, 0), -1)
                cv2.putText(combined, "CORRECT! Great job!",
                            (20, combined.shape[0] - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                cv2.imshow("Signify AI - Learn", combined)
                cv2.waitKey(1200)
                return True

            cv2.imshow("Signify AI - Learn", combined)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    return False


def learn_mode(model):
    labels = trained_labels(model)

    print("\n=== LEARN SIGN LANGUAGE ===")
    print("Type a word/sign you want to learn.")
    print("If the whole word is trained, Signify AI shows its reference pose.")
    print("Otherwise it tries to teach the word letter-by-letter.")

    text = input("\nWhat do you want to learn? > ").strip()
    if not text:
        return

    whole = normalize_label(text)

    if whole in labels:
        practice_sign(model, whole)
        input("\nPress Enter to return...")
        return

    letters = [c for c in text.upper() if c.isalpha()]
    missing = sorted({c for c in letters if c not in labels})

    if not letters:
        print("Please enter a word.")
    elif missing:
        print("\nThe whole word is not trained.")
        print("Missing trained letters:", ", ".join(missing))
        print("\nTrain the whole word with:")
        print(f'  python collect_data.py "{whole}"')
        print("or train the missing letters, then run python train_model.py.")
    else:
        print("\nFingerspelling:", " -> ".join(letters))
        input("Press Enter to start...")
        for i, letter in enumerate(letters, 1):
            print(f"\nLetter {i}/{len(letters)}: {letter}")
            if not practice_sign(model, letter):
                break

    input("\nPress Enter to return...")


def test_question(model, landmarker, cap, target, number, total):
    target = normalize_label(target)
    correct_frames = 0
    started = time.monotonic()
    limit = 20

    while True:
        ok, frame = cap.read()
        if not ok:
            return "quit"

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(
            image, int(time.monotonic() * 1000)
        )

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)
            predicted, confidence, _ = predict_hand(model, hand)
            if normalize_label(predicted) == target and confidence >= CONFIDENCE_THRESHOLD:
                correct_frames += 1
            else:
                correct_frames = 0
        else:
            correct_frames = 0

        remaining = max(0, int(limit - (time.monotonic() - started)))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 115), (0, 0, 0), -1)
        cv2.putText(frame, f"Question {number}/{total}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 200, 200), 2)
        cv2.putText(frame, f"MAKE: {target.replace('_', ' ')}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {remaining}s   S=skip   Q=quit", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if correct_frames >= HOLD_FRAMES:
            cv2.putText(frame, "CORRECT! +1", (20, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow("Signify AI - Test", frame)
            cv2.waitKey(900)
            return "correct"

        cv2.imshow("Signify AI - Test", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "skip"
        if remaining <= 0:
            return "timeout"


def test_mode(model):
    labels = [str(x) for x in model.classes_]

    if len(labels) < 2:
        print("\nTrain at least two signs first.")
        input("Press Enter to return...")
        return

    raw = input("\nHow many test questions? [5] > ").strip()
    try:
        total = int(raw) if raw else 5
    except ValueError:
        total = 5
    total = max(1, min(total, 20))

    landmarker = create_landmarker()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        landmarker.close()
        print("Could not open webcam.")
        return

    score = 0
    attempted = 0

    try:
        for number in range(1, total + 1):
            target = random.choice(labels)
            result = test_question(model, landmarker, cap, target, number, total)
            if result == "quit":
                break
            attempted += 1
            if result == "correct":
                score += 1
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

    print("\n=== TEST RESULT ===")
    if attempted:
        percentage = score / attempted * 100
        print(f"Score: {score}/{attempted} ({percentage:.0f}%)")
        if percentage >= 80:
            print("Excellent!")
        elif percentage >= 60:
            print("Good job. Keep practicing!")
        else:
            print("Use Learn Mode and try again.")
    else:
        print("No questions completed.")

    input("\nPress Enter to return...")


def main():
    model = load_model()
    if model is None:
        return

    while True:
        print("\n" + "=" * 56)
        print("SIGNIFY AI - AI SIGN LANGUAGE HELPER")
        print("=" * 56)
        print("1. Learn Sign Language")
        print("2. Take a Test")
        print("Q. Exit")

        choice = input("\nWhat do you want to do? > ").strip().lower()

        if choice in {"1", "learn", "learn sign language", "learn sign lang"}:
            learn_mode(model)
        elif choice in {"2", "test", "take a test", "take test"}:
            test_mode(model)
        elif choice in {"q", "quit", "exit", "3"}:
            break
        else:
            print("Type 1/learn, 2/test, or Q/exit.")


if __name__ == "__main__":
    main()
