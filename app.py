import csv
import os
import random
import time
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("GLOG_logtostderr", "0")

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

from hand_utils import HAND_CONNECTIONS, create_landmarker, draw_hand, extract_features

MODEL_FILE = Path("models/sign_classifier.joblib")
DATA_FILE = Path("data/signs.csv")
SIGN_IMAGE_DIR = Path("assets/signs")

CONFIDENCE_THRESHOLD = 0.60
HOLD_FRAMES = 6
WAVE_MOTION_THRESHOLD = 0.015


# -----------------------------
# Small helpers
# -----------------------------

def normalize_label(text):
    return "_".join(text.strip().upper().replace("-", " ").split())


def pretty_label(text):
    return str(text).replace("_", " ")


def open_fullscreen(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
    except cv2.error:
        pass


def smooth_prediction(history):
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]


def wrist_motion_score(wrist_history):
    if len(wrist_history) < 2:
        return 0.0

    points = np.asarray(wrist_history, dtype=np.float32)
    movement = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.mean(movement))


def load_model():
    if not MODEL_FILE.exists():
        print("\nTrained model not found.")
        print("Collect at least two signs and run: python train_model.py")
        return None

    return joblib.load(MODEL_FILE)


def trained_labels(model):
    return {
        normalize_label(str(label)): str(label)
        for label in model.classes_
    }


def predict_hand(model, hand):
    features = extract_features(hand)

    if hasattr(model, "feature_names_in_"):
        sample = pd.DataFrame(
            [features],
            columns=model.feature_names_in_,
            dtype=np.float32,
        )
        probabilities = model.predict_proba(sample)[0]
    else:
        probabilities = model.predict_proba(features.reshape(1, -1))[0]

    best_index = int(np.argmax(probabilities))
    label = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    return label, confidence, features


def detect_hand(landmarker, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(image, int(time.monotonic() * 1000))


# -----------------------------
# Learn mode helpers
# -----------------------------

def load_reference_pose(label):
    if not DATA_FILE.exists():
        return None

    wanted = normalize_label(label)
    rows = []
    feature_names = [f"f{i}" for i in range(63)]

    with DATA_FILE.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if normalize_label(row.get("label", "")) != wanted:
                continue

            try:
                rows.append([float(row[name]) for name in feature_names])
            except (KeyError, TypeError, ValueError):
                continue

    if not rows:
        return None

    average = np.mean(np.asarray(rows, dtype=np.float32), axis=0)
    return average.reshape(21, 3)


def find_sign_image(label):
    name = normalize_label(label)

    for extension in (".png", ".jpg", ".jpeg", ".webp"):
        path = SIGN_IMAGE_DIR / f"{name}{extension}"
        if path.exists():
            return path

    return None


def draw_photo_guide(panel, label, image_path):
    photo = cv2.imread(str(image_path))
    if photo is None:
        return False

    height, width = panel.shape[:2]

    cv2.putText(
        panel,
        "How to make this sign",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        pretty_label(label),
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
    )

    max_width = width - 30
    max_height = max(height - 135, 80)
    photo_height, photo_width = photo.shape[:2]

    scale = min(max_width / photo_width, max_height / photo_height)
    new_width = max(1, int(photo_width * scale))
    new_height = max(1, int(photo_height * scale))

    photo = cv2.resize(
        photo,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    x = (width - new_width) // 2
    y = 90 + max(0, (max_height - new_height) // 2)
    panel[y:y + new_height, x:x + new_width] = photo

    cv2.putText(
        panel,
        "Try to copy this hand shape",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
    )

    return True


def draw_pose_guide(panel, label, pose):
    height, width = panel.shape[:2]

    cv2.putText(
        panel,
        "Practice guide",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        pretty_label(label),
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
    )

    if pose is None:
        cv2.putText(
            panel,
            "No photo or reference available",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
        )
        return

    points = pose[:, :2]
    extent = max(float(np.max(np.abs(points))), 1e-6)
    usable_height = max(height - 160, 120)
    scale = min(width - 70, usable_height) * 0.48 / extent

    center_x = width // 2
    center_y = 110 + usable_height // 2

    pixel_points = [
        (int(center_x + x * scale), int(center_y + y * scale))
        for x, y in points
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(panel, pixel_points[start], pixel_points[end], (255, 255, 255), 3)

    for point in pixel_points:
        cv2.circle(panel, point, 6, (0, 255, 0), -1)

    cv2.putText(
        panel,
        "Match your hand to this guide",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
    )


def reference_panel(label, pose, height):
    panel = np.zeros((height, 330, 3), dtype=np.uint8)
    image_path = find_sign_image(label)

    if image_path and draw_photo_guide(panel, label, image_path):
        return panel

    draw_pose_guide(panel, label, pose)
    return panel


def pose_similarity(features, reference):
    if reference is None:
        return None

    current = np.asarray(features, dtype=np.float32).reshape(21, 3)
    distance = float(
        np.mean(
            np.linalg.norm(
                current[:, :2] - reference[:, :2],
                axis=1,
            )
        )
    )

    return max(0.0, min(100.0, 100.0 * (1.0 - distance / 0.45)))


def practice_feedback(similarity):
    if similarity is None:
        return "Show your hand clearly"
    if similarity >= 90:
        return "Excellent match!"
    if similarity >= 75:
        return "Very close - small adjustment"
    if similarity >= 55:
        return "Good try - adjust your fingers a little"
    return "Keep going - match the guide more closely"


def practice_sign(model, target):
    target = normalize_label(target)

    if target not in trained_labels(model):
        return False

    reference = load_reference_pose(target)
    landmarker = create_landmarker()
    camera = cv2.VideoCapture(0)
    window_name = "Signify AI - Learn"

    if not camera.isOpened():
        landmarker.close()
        print("Could not open webcam.")
        return False

    open_fullscreen(window_name)

    history = deque(maxlen=8)
    wrist_history = deque(maxlen=6)
    correct_frames = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = detect_hand(landmarker, frame)

            label = "No hand"
            confidence = 0.0
            similarity = None
            motion = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)

                wrist_history.append((hand[0].x, hand[0].y))
                motion = wrist_motion_score(wrist_history)

                raw_label, confidence, features = predict_hand(model, hand)
                similarity = pose_similarity(features, reference)

                if confidence >= CONFIDENCE_THRESHOLD:
                    history.append(raw_label)
                    label = smooth_prediction(history) or raw_label
                else:
                    history.clear()
                    label = "Not sure"

                if (
                    normalize_label(label) == target
                    and confidence >= CONFIDENCE_THRESHOLD
                ):
                    correct_frames += 1
                else:
                    correct_frames = 0
            else:
                history.clear()
                wrist_history.clear()
                correct_frames = 0

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 145), (0, 0, 0), -1)

            cv2.putText(
                frame,
                f"Let's learn: {pretty_label(target)}",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"I can see: {pretty_label(label)}",
                (15, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (0, 255, 255),
                2,
            )

            if similarity is not None:
                cv2.putText(
                    frame,
                    f"Match: {similarity:.0f}%",
                    (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                )

            feedback = practice_feedback(similarity)
            if motion >= WAVE_MOTION_THRESHOLD:
                feedback += "  |  Movement detected"

            cv2.putText(
                frame,
                feedback,
                (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
            )

            combined = np.hstack(
                [frame, reference_panel(target, reference, frame.shape[0])]
            )

            if correct_frames >= HOLD_FRAMES:
                cv2.rectangle(
                    combined,
                    (0, combined.shape[0] - 70),
                    (combined.shape[1], combined.shape[0]),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    combined,
                    "That's right! Great job!",
                    (20, combined.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )
                cv2.imshow(window_name, combined)
                cv2.waitKey(1200)
                return True

            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()

    return False


def learn_mode(model):
    labels = trained_labels(model)

    print("\n=== LEARN SIGN LANGUAGE ===")
    print("Enter a trained sign, letter or word.")

    text = input("\nWhat do you want to learn? > ").strip()
    if not text:
        return

    whole = normalize_label(text)

    if whole in labels:
        practice_sign(model, whole)
        input("\nPress Enter to return...")
        return

    letters = [char for char in text.upper() if char.isalpha()]
    missing = sorted({letter for letter in letters if letter not in labels})

    if not letters:
        print("Please enter a word or letter.")
    elif missing:
        print("\nThat whole sign is not trained yet.")
        print("Missing letters:", ", ".join(missing))
        print("\nYou can train the whole word with:")
        print(f'  python collect_data.py "{whole}"')
        print("or train the missing letters and run python train_model.py again.")
    else:
        print("\nFingerspelling:", " -> ".join(letters))
        input("Press Enter to start...")

        for number, letter in enumerate(letters, 1):
            print(f"\nLetter {number}/{len(letters)}: {letter}")
            if not practice_sign(model, letter):
                break

    input("\nPress Enter to return...")


# -----------------------------
# Test mode
# -----------------------------

def test_question(model, landmarker, camera, target, number, total, window_name):
    target = normalize_label(target)
    correct_frames = 0
    started = time.monotonic()
    time_limit = 20

    while True:
        ok, frame = camera.read()
        if not ok:
            return "quit"

        frame = cv2.flip(frame, 1)
        result = detect_hand(landmarker, frame)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            draw_hand(frame, hand)

            predicted, confidence, _ = predict_hand(model, hand)

            if (
                normalize_label(predicted) == target
                and confidence >= CONFIDENCE_THRESHOLD
            ):
                correct_frames += 1
            else:
                correct_frames = 0
        else:
            correct_frames = 0

        elapsed = time.monotonic() - started
        remaining = max(0, int(time_limit - elapsed))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 115), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Question {number}/{total}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (200, 200, 200),
            2,
        )
        cv2.putText(
            frame,
            f"Show me: {pretty_label(target)}",
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Time left: {remaining}s   S=skip   Q=quit",
            (15, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
        )

        if correct_frames >= HOLD_FRAMES:
            cv2.putText(
                frame,
                "Correct! Well done!",
                (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3,
            )
            cv2.imshow(window_name, frame)
            cv2.waitKey(900)
            return "correct"

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            return "quit"
        if key == ord("s"):
            return "skip"
        if remaining <= 0:
            return "timeout"


def custom_test_targets(text, model):
    labels = trained_labels(model)
    whole = normalize_label(text)

    if whole in labels:
        return [labels[whole]], []

    pieces = [piece for piece in text.replace(",", " ").split() if piece]

    if len(pieces) > 1:
        targets = []
        missing = []

        for piece in pieces:
            key = normalize_label(piece)
            if key in labels:
                targets.append(labels[key])
            else:
                missing.append(piece.upper())

        return targets, missing

    letters = [char for char in text.upper() if char.isalpha()]
    targets = []
    missing = []

    for letter in letters:
        if letter in labels:
            targets.append(labels[letter])
        else:
            missing.append(letter)

    return targets, sorted(set(missing))


def test_mode(model):
    labels = [str(label) for label in model.classes_]

    if len(labels) < 2:
        print("\nTrain at least two signs first.")
        input("Press Enter to return...")
        return

    print("\n=== TAKE A TEST ===")
    print("1. Random test")
    print("2. Test a specific letter or word")

    mode = input("\nChoose > ").strip().lower()

    if mode in {"2", "custom", "specific"}:
        text = input("Enter A, ABC, HELLO, A B C, etc. > ").strip()
        targets, missing = custom_test_targets(text, model)

        if missing:
            print("\nThese signs are not trained:", ", ".join(missing))
            input("\nPress Enter to return...")
            return

        if not targets:
            print("No valid trained signs found.")
            input("\nPress Enter to return...")
            return
    else:
        raw = input("How many random questions? [5] > ").strip()

        try:
            total = int(raw) if raw else 5
        except ValueError:
            total = 5

        total = max(1, min(total, 20))
        targets = [random.choice(labels) for _ in range(total)]

    landmarker = create_landmarker()
    camera = cv2.VideoCapture(0)
    window_name = "Signify AI - Test"

    if not camera.isOpened():
        landmarker.close()
        print("Could not open webcam.")
        return

    open_fullscreen(window_name)

    score = 0
    attempted = 0

    try:
        for number, target in enumerate(targets, 1):
            result = test_question(
                model,
                landmarker,
                camera,
                target,
                number,
                len(targets),
                window_name,
            )

            if result == "quit":
                break

            attempted += 1

            if result == "correct":
                score += 1

    finally:
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()

    print("\n=== TEST RESULT ===")

    if not attempted:
        print("No questions completed.")
        input("\nPress Enter to return...")
        return

    percentage = score / attempted * 100
    print(f"Score: {score}/{attempted} ({percentage:.0f}%)")

    if percentage >= 80:
        print("Excellent!")
    elif percentage >= 60:
        print("Good job. Keep practicing!")
    else:
        print("Try Learn Mode once more and then test yourself again.")

    input("\nPress Enter to return...")


# -----------------------------
# Identifier mode
# -----------------------------

def identify_mode(model):
    landmarker = create_landmarker()
    camera = cv2.VideoCapture(0)
    window_name = "Signify AI - Identifier"

    if not camera.isOpened():
        landmarker.close()
        print("Could not open webcam.")
        return

    open_fullscreen(window_name)

    history = deque(maxlen=8)
    wrist_history = deque(maxlen=6)
    last_label = None
    sequence = []

    print("\nIdentify mode started.")
    print("Q = return   SPACE = add sign   C = clear")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = detect_hand(landmarker, frame)

            label = "No hand"
            confidence = 0.0
            motion = 0.0

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)

                wrist_history.append((hand[0].x, hand[0].y))
                motion = wrist_motion_score(wrist_history)

                raw_label, confidence, _ = predict_hand(model, hand)

                if confidence >= CONFIDENCE_THRESHOLD:
                    history.append(raw_label)
                    label = smooth_prediction(history) or raw_label
                    last_label = label
                else:
                    history.clear()
                    label = "Not sure"
                    last_label = None
            else:
                history.clear()
                wrist_history.clear()
                last_label = None

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 145), (0, 0, 0), -1)

            cv2.putText(
                frame,
                "Signify AI - Live Sign Reader",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"I can see: {pretty_label(label)}",
                (15, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.90,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (15, 102),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

            movement_text = (
                "Movement detected"
                if motion >= WAVE_MOTION_THRESHOLD
                else "Hold your hand clearly"
            )

            cv2.putText(
                frame,
                movement_text,
                (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                1,
            )

            sequence_text = " ".join(sequence[-8:]) or "-"

            cv2.rectangle(
                frame,
                (0, frame.shape[0] - 75),
                (frame.shape[1], frame.shape[0]),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame,
                f"Your signs: {sequence_text}",
                (15, frame.shape[0] - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                frame,
                "SPACE: add   C: clear   Q: back",
                (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("c"):
                sequence.clear()

            if key == 32 and last_label:
                sequence.append(str(last_label))
                history.clear()
                last_label = None

    finally:
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


# -----------------------------
# Main menu
# -----------------------------

def main():
    model = load_model()
    if model is None:
        return

    while True:
        print("\n" + "=" * 48)
        print("SIGNIFY AI")
        print("AI Sign Language Helper")
        print("=" * 48)
        print("1. Learn Sign Language")
        print("2. Take a Test")
        print("3. Identify a Sign")
        print("Q. Exit")

        choice = input("\nWhat do you want to do? > ").strip().lower()

        if choice in {"1", "learn", "learn sign language", "learn sign lang"}:
            learn_mode(model)
        elif choice in {"2", "test", "take a test", "take test"}:
            test_mode(model)
        elif choice in {"3", "identify", "identifier", "identify sign"}:
            identify_mode(model)
        elif choice in {"q", "quit", "exit", "4"}:
            break
        else:
            print("Choose 1, 2, 3 or Q.")


if __name__ == "__main__":
    main()