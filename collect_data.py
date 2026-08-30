import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp

from hand_utils import create_landmarker, extract_features, draw_hand

DATA_FILE = Path("data/signs.csv")
FEATURE_COUNT = 63

def append_sample(label, features):
    DATA_FILE.parent.mkdir(exist_ok=True)
    new_file = not DATA_FILE.exists()

    with DATA_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["label"] + [f"f{i}" for i in range(FEATURE_COUNT)])
        writer.writerow([label] + list(map(float, features)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", help='Name of the sign, e.g. HELLO or YES')
    parser.add_argument("--samples", type=int, default=250)
    args = parser.parse_args()

    label = args.label.strip().upper()
    target = args.samples

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    landmarker = create_landmarker()

    collecting = False
    count = 0
    last_saved = 0.0

    print(f"Label: {label}")
    print("Press C to start/stop collecting.")
    print("Move/rotate your hand slightly while collecting.")
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

            hand_found = bool(result.hand_landmarks)

            if hand_found:
                hand = result.hand_landmarks[0]
                draw_hand(frame, hand)

                if collecting and count < target:
                    now = time.monotonic()
                    if now - last_saved >= 0.07:
                        features = extract_features(hand)
                        append_sample(label, features)
                        count += 1
                        last_saved = now

                        if count >= target:
                            collecting = False
                            print(f"Finished: saved {count} samples for {label}")

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
            cv2.putText(
                frame, f"Sign: {label}   Samples: {count}/{target}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
            )
            status = "COLLECTING" if collecting else "Press C to collect"
            color = (0, 255, 0) if collecting else (0, 200, 255)
            cv2.putText(
                frame, status, (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            if not hand_found:
                cv2.putText(
                    frame, "Show your hand to the camera",
                    (230, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
                )

            cv2.imshow("Collect Sign Language Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("c"):
                collecting = not collecting

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
