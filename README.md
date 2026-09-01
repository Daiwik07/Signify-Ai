# Signify AI

Signify AI is a real-time hand-sign recognition project built with Python, OpenCV, MediaPipe, and a Random Forest classifier.

It can:

- Learn and practice trained hand signs
- Show a reference image while learning
- Test the user with trained signs
- Identify signs live through the webcam
- Show confidence and pose similarity
- Run in full-screen mode for presentation/demo

## Required Files for Competition Submission

```text
Signify-Ai/
│
├── app.py
├── hand_utils.py
├── requirements.txt
│
├── assets/
│   └── signs/
│       ├── A.png
│       ├── B.png
│       ├── C.png
│       └── ...
│
├── data/
│   └── signs.csv
│
└── models/
    ├── sign_classifier.joblib
    └── hand_landmarker.task
```

### What each file does

- `app.py` — main Signify AI application
- `hand_utils.py` — hand detection, landmark extraction, and drawing functions
- `requirements.txt` — Python packages needed to run the project
- `assets/signs/` — reference sign images used in Learn mode
- `data/signs.csv` — collected landmark data used for reference poses and similarity
- `models/sign_classifier.joblib` — trained sign-classification model
- `models/hand_landmarker.task` — MediaPipe hand landmark model

`collect_data.py` and `train_model.py` are development/training files and are not required just to run the finished competition demo.

## Requirements

Recommended Python version: **Python 3.11**

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python app.py
```

The main menu provides:

```text
1. Learn Sign Language
2. Take a Test
3. Identify a Sign
Q. Exit
```

## How It Works

The webcam image is processed by MediaPipe, which detects 21 hand landmarks. These landmarks are converted into 63 normalized values and passed to the trained Random Forest model to recognize the sign.

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
63 Normalized Features
   ↓
Random Forest Classifier
   ↓
Recognized Sign
```

## Note

Signify AI currently focuses mainly on trained hand shapes. Complete sign languages can also depend on movement, two hands, facial expressions, body position, and grammar.

## Project Goal

The goal of Signify AI is to demonstrate how computer vision and machine learning can make learning and recognizing hand signs more interactive and accessible.