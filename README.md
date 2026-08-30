# AI Sign Language Helper (Python)

A beginner-friendly AI project that learns hand signs from your webcam and recognizes them in real time.

## What it uses

- Python
- OpenCV for the webcam
- MediaPipe Hand Landmarker for 21 hand landmarks
- scikit-learn Random Forest for sign classification

## Important

This starter project recognizes **static hand signs that you train yourself**.
Real sign languages also use motion, two hands, facial expressions, body position, and grammar.
So do not claim that this starter version translates an entire sign language.

## 1. Install

Recommended: Python 3.11.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

The MediaPipe hand model downloads automatically the first time you run the program.

## 2. Collect training examples

Collect around 250 samples for each sign.

Example:

```bash
python collect_data.py HELLO
python collect_data.py YES
python collect_data.py NO
python collect_data.py THANK_YOU
```

When the webcam opens:

- Press `C` to start collecting.
- Keep making the same sign.
- Move the hand slightly closer/farther and rotate it a little.
- Press `C` again if you want to pause.
- Press `Q` to quit.

For better accuracy, record the sign under different lighting and backgrounds.

## 3. Train the AI

```bash
python train_model.py
```

The trained classifier is saved to:

```text
models/sign_classifier.joblib
```

## 4. Run the helper

```bash
python app.py
```

Show one of the trained signs to the webcam.

Press `Q` to exit.

## Good school-demo features to add next

1. Learning mode: show a word and ask the user to make the correct sign.
2. Score system: +10 points for a correct sign.
3. Text-to-speech: speak the recognized word.
4. Sentence builder: save recognized signs as words.
5. Two-hand recognition.
6. Dynamic-sign recognition using a short sequence of frames and an LSTM/Transformer.
7. Indian Sign Language dataset and ISL-specific labels.

## Suggested presentation line

> Our AI does not try to replace sign-language interpreters. It demonstrates how computer vision can make sign-language learning and basic communication more accessible by recognizing trained hand signs in real time.
