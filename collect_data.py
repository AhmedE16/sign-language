"""
collect_data.py  (MediaPipe Tasks API — compatible with Python 3.13 + mediapipe 0.10.x)
───────────────
Interactive dataset collector for sign language recognition.

Controls (while camera window is open):
    s  ─ Start / Stop collecting frames for the current label
    n  ─ Enter a new label (prompted in terminal)
    d  ─ Delete last saved session for current label
    q  ─ Quit and save CSV
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import os
import sys
import urllib.request

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "dataset.csv"
MODEL_FILE  = "hand_landmarker.task"
MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
SKIP_NO_HAND             = True
MIN_DETECTION_CONFIDENCE = 0.7
MIN_PRESENCE_CONFIDENCE  = 0.7
MIN_TRACKING_CONFIDENCE  = 0.5
# ──────────────────────────────────────────────────────────────────────────────

LANDMARK_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
ALL_COLS      = LANDMARK_COLS + ["label"]

# Hand skeleton connections for manual drawing
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


def download_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Downloading hand landmark model (~2 MB) …")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
            print(f"✔  Model saved to {MODEL_FILE}")
        except Exception as e:
            sys.exit(f"❌  Failed to download model: {e}\n"
                     f"    Download manually from:\n    {MODEL_URL}")


def init_landmarker():
    download_model()
    base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_FILE)
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
    """Return flat list of 63 floats or None if no hand found."""
    if result.hand_landmarks:
        lms = result.hand_landmarks[0]   # first hand
        return [coord for lm in lms for coord in (lm.x, lm.y, lm.z)]
    return None


def draw_hand(frame, result):
    if not result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    lms  = result.hand_landmarks[0]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 120, 255), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 210, 90), -1)


def draw_overlay(frame, label, collecting, frame_count, session_count):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    status_color = (0, 220, 80) if collecting else (60, 60, 200)
    status_text  = "● RECORDING" if collecting else "○ PAUSED"
    cv2.putText(frame, status_text, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    label_str = f"Label: [{label}]" if label else "Label: [none — press N]"
    cv2.putText(frame, label_str, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    info = f"Frames: {frame_count}  |  Sessions: {session_count}"
    cv2.putText(frame, info, (w - 280, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

    cv2.rectangle(frame, (0, h - 30), (w, h), (30, 30, 30), -1)
    legend = "S: Start/Stop  |  N: New label  |  D: Delete last  |  Q: Quit & save"
    cv2.putText(frame, legend, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def ask_label(current):
    print(f"\nCurrent label: '{current}'")
    new = input("Enter new label (letter / word): ").strip()
    return new if new else current


def main():
    landmarker = init_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("❌  Cannot open webcam.")

    # Load or create CSV
    if os.path.exists(CSV_PATH):
        df_all = pd.read_csv(CSV_PATH)
        print(f"✔  Loaded existing dataset: {len(df_all)} rows  →  {CSV_PATH}")
    else:
        df_all = pd.DataFrame(columns=ALL_COLS)
        print(f"✔  Starting new dataset  →  {CSV_PATH}")

    current_label   = ""
    collecting      = False
    session_rows    = []
    session_count   = 0
    session_history = []

    print("\n📷  Camera open. Press N to set a label, then S to record.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        draw_hand(frame, result)

        if collecting and current_label:
            lm = extract_landmarks(result)
            if lm is not None:
                session_rows.append(lm + [current_label])
            elif not SKIP_NO_HAND:
                session_rows.append([np.nan] * 63 + [current_label])

        draw_overlay(frame, current_label, collecting,
                     len(session_rows), session_count)
        cv2.imshow("Sign Language – Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            if not current_label:
                print("⚠  Set a label first (press N).")
                continue
            if collecting:
                collecting = False
                if session_rows:
                    new_df   = pd.DataFrame(session_rows, columns=ALL_COLS)
                    df_all   = pd.concat([df_all, new_df], ignore_index=True)
                    session_history.append((current_label, len(session_rows)))
                    session_count += 1
                    print(f"✔  Saved {len(session_rows)} frames for '{current_label}'  "
                          f"(total: {len(df_all)})")
                else:
                    print("⚠  No frames collected (was a hand visible?).")
                session_rows = []
            else:
                collecting   = True
                session_rows = []
                print(f"🔴  Recording '{current_label}' …  press S to stop")

        elif key == ord("n"):
            was_collecting = collecting
            collecting     = False
            cv2.destroyAllWindows()
            current_label  = ask_label(current_label)
            print(f"✔  Label set to '{current_label}'")
            collecting     = was_collecting

        elif key == ord("d"):
            if session_history:
                last_label, last_count = session_history.pop()
                df_all        = df_all.iloc[:-last_count].reset_index(drop=True)
                session_count -= 1
                print(f"🗑  Deleted {last_count} rows of '{last_label}'  "
                      f"(remaining: {len(df_all)})")
            else:
                print("⚠  Nothing to undo.")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if not df_all.empty:
        df_all.to_csv(CSV_PATH, index=False)
        print(f"\n💾  Saved {len(df_all)} rows  →  {CSV_PATH}")
        print("\nFrames per label:")
        print(df_all["label"].value_counts().to_string())
    else:
        print("\n⚠  No data collected – CSV not written.")


if __name__ == "__main__":
    main()
