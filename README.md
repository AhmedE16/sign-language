# 🤟 Real-Time Sign Language Recognition System

A real-time hand sign language recognition pipeline built with **MediaPipe Hand Landmarker** and **Scikit-learn**. Collects custom hand landmark datasets via webcam, trains a machine learning classifier, and performs live inference — all from scratch.

---

## 📌 Demo Pipeline

```
Webcam → MediaPipe Hand Landmarks → CSV Dataset → ML Classifier → Real-Time Prediction
```

---

## 🗂️ Project Structure

```
sign-language-recognition/
├── collect_data.py        # Step 1 — interactive dataset collection via webcam
├── train_model.py         # Step 2 — train Random Forest / SVM classifier
├── realtime_predict.py    # Step 3 — live webcam inference with smoothing
├── requirements.txt
├── hand_landmarker.task   # auto-downloaded on first run (~2 MB)
├── dataset.csv            # generated after Step 1
├── model.pkl              # generated after Step 2
└── label_map.pkl          # generated after Step 2
```

---

## ⚙️ How It Works

### 1. Hand Landmark Extraction
MediaPipe's **Hand Landmarker** model detects 21 3D keypoints on the hand. Each frame is represented as a flat vector of **63 features** (x, y, z × 21 joints), normalized relative to the image frame — making it robust to hand scale and position.

```
Landmark 0 (Wrist) → (x0, y0, z0)
Landmark 1–4       → Thumb joints
Landmark 5–8       → Index finger
...
Landmark 17–20     → Pinky finger
```

### 2. Dataset Collection (`collect_data.py`)
- Opens webcam and lets you record signs interactively
- Each frame where a hand is detected → one row in `dataset.csv`
- Supports unlimited labels (letters, words, phrases)
- Undo last session, append across multiple runs

### 3. Model Training (`train_model.py`)
- Loads `dataset.csv`, drops NaN rows (no-hand frames)
- Encodes labels with `LabelEncoder`
- Trains a **Random Forest** (or SVM) inside a `StandardScaler` pipeline
- Outputs classification report, confusion matrix, and 5-fold CV score

### 4. Real-Time Prediction (`realtime_predict.py`)
- Extracts landmarks per frame and feeds them to the trained model
- **Smoothing buffer** (majority vote over 15 frames) prevents flickering
- Confidence threshold filters low-certainty predictions
- Auto-logs stable predictions (held for 20 frames) to a history strip

---
| Key | Action |
|-----|--------|
| `N` | Enter a new label (e.g. `A`, `hello`, `yes`) |
| `S` | Toggle recording ON / OFF |
| `D` | Undo last session |
| `Q` | Quit and save `dataset.csv` |

Recorded 10–15 seconds per label (~300–450 frames). Slightly vary hand distance and tilt while keeping the sign correct.


## 📊 Data Format

Each row in `dataset.csv`:

| x0 | y0 | z0 | x1 | y1 | z1 | … | x20 | y20 | z20 | label |
|----|----|----|----|----|----|----|-----|-----|-----|-------|
| 0.51 | 0.73 | 0.01 | … | … | … | … | … | … | … | A |

63 landmark features + 1 label column.

---

## 🧰 Tech Stack

| Component | Library |
|-----------|---------|
| Hand Landmark Detection | MediaPipe Tasks API (Hand Landmarker) |
| Computer Vision | OpenCV |
| Machine Learning | Scikit-learn (Random Forest / SVM) |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |

---

## for Better Accuracy

- Aim for **300+ frames per label** for a solid model
- Include variation: hand distance, slight rotation, different lighting
- Check `confusion_matrix.png` after training to identify confused pairs
- Switch to SVM (`CLASSIFIER = "svm"` in `train_model.py`) for smaller datasets
- Add more data anytime — `collect_data.py` appends to existing CSV

---

## 🔧 Requirements

```
opencv-python>=4.8
mediapipe>=0.10
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
joblib>=1.3
matplotlib>=3.7
```

---



## 👤 Author

**Ahmad Essam**  
Computer & Systems Engineering — Ain Shams University  
[LinkedIn](https://linkedin.com/in/ahmed-essam-eng) 
