"""
realtime_predict.py  (MediaPipe Tasks API — compatible with Python 3.13 + mediapipe 0.10.x)
───────────────────
Live sign-language prediction using the trained model.

Controls:
    q  ─ Quit
    c  ─ Clear prediction history
"""

import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import os
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH               = "model.pkl"
LABEL_MAP_PATH           = "label_map.pkl"
HAND_MODEL_FILE          = "hand_landmarker.task"
SMOOTHING_WINDOW         = 15
CONFIDENCE_THRESH        = 0.60
STABLE_NEEDED            = 20
MIN_DETECTION_CONFIDENCE = 0.7
MIN_PRESENCE_CONFIDENCE  = 0.7
MIN_TRACKING_CONFIDENCE  = 0.5
# ──────────────────────────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def load_ml_artifacts():
    for path in (MODEL_PATH, LABEL_MAP_PATH):
        if not os.path.exists(path):
            sys.exit(f"❌  {path} not found. Run train_model.py first.")
    return joblib.load(MODEL_PATH), joblib.load(LABEL_MAP_PATH)


def init_landmarker():
    if not os.path.exists(HAND_MODEL_FILE):
        sys.exit(f"❌  {HAND_MODEL_FILE} not found.\n"
                 "    Run collect_data.py first — it downloads the model automatically.")
    base_options = mp_tasks.BaseOptions(model_asset_path=HAND_MODEL_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(result):
    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        return np.array([coord for lm in lms for coord in (lm.x, lm.y, lm.z)],
                        dtype=np.float32)
    return None


def draw_hand(frame, result):
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    lms  = result.hand_landmarks[0]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (60, 140, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 210, 90), -1)


def majority_vote(buffer, le):
    if not buffer:
        return "…", 0.0
    from collections import Counter
    idx, count = Counter(buffer).most_common(1)[0]
    return le.inverse_transform([idx])[0], count / len(buffer)


def draw_ui(frame, label, confidence, hand_found, history):
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 70), (25, 25, 25), -1)

    if hand_found:
        col  = (0, 220, 80) if confidence >= CONFIDENCE_THRESH else (0, 200, 220)
        text = label
    else:
        col  = (140, 140, 140)
        text = "No hand detected"

    cv2.putText(frame, text, (15, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, col, 2, cv2.LINE_AA)

    if hand_found:
        conf_str = f"{confidence * 100:.1f}%"
        cv2.putText(frame, conf_str, (w - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1)

        bar_x, bar_y, bar_w, bar_h = 15, 62, w - 30, 5
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60,60,60), -1)
        fill    = int(bar_w * confidence)
        bar_col = (0, 220, 80) if confidence >= CONFIDENCE_THRESH else (0, 200, 220)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_col, -1)

    cv2.rectangle(frame, (0, h - 36), (w, h), (25, 25, 25), -1)
    hist_str = "History: " + "  ".join(list(history)[-14:])
    cv2.putText(frame, hist_str, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)
    cv2.putText(frame, "Q: Quit   C: Clear", (w - 200, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)


def main():
    print("Loading classifier …")
    model, le = load_ml_artifacts()
    print(f"✔  Classes: {list(le.classes_)}")

    landmarker  = init_landmarker()
    cap         = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("❌  Cannot open webcam.")

    smooth_buf  = deque(maxlen=SMOOTHING_WINDOW)
    conf_buf    = deque(maxlen=SMOOTHING_WINDOW)
    history     = deque(maxlen=40)
    last_stable = None
    stable_run  = 0

    print("📷  Live prediction running. Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        draw_hand(frame, result)
        lm = extract_landmarks(result)

        if lm is not None:
            proba    = model.predict_proba([lm])[0]
            pred_idx = int(np.argmax(proba))
            pred_conf = float(proba[pred_idx])
            smooth_buf.append(pred_idx)
            conf_buf.append(pred_conf)

            smooth_label, smooth_conf = majority_vote(smooth_buf, le)
            avg_conf = float(np.mean(conf_buf))

            if smooth_label == last_stable:
                stable_run += 1
            else:
                stable_run  = 1
                last_stable = smooth_label

            if stable_run == STABLE_NEEDED and smooth_conf >= CONFIDENCE_THRESH:
                history.append(smooth_label)
                print(f"  → {smooth_label}  ({avg_conf * 100:.1f}%)")

            draw_ui(frame, smooth_label, avg_conf, True, history)
        else:
            smooth_buf.clear()
            conf_buf.clear()
            draw_ui(frame, "—", 0.0, False, history)

        cv2.imshow("Sign Language – Live Prediction", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("c"):
            history.clear()
            smooth_buf.clear()
            conf_buf.clear()
            last_stable = None
            print("  History cleared.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Done.")


if __name__ == "__main__":
    main()
