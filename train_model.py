"""
train_model.py
──────────────
Train a Random Forest classifier on the landmark CSV produced by collect_data.py.

Usage:
    python train_model.py

Outputs:
    model.pkl      ─ trained classifier  (joblib)
    label_map.pkl  ─ label encoder       (joblib)
    report.txt     ─ classification report
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt

from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH       = "dataset.csv"
MODEL_PATH     = "model.pkl"
LABEL_MAP_PATH = "label_map.pkl"
REPORT_PATH    = "report.txt"
CM_PATH        = "confusion_matrix.png"

TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# Choose classifier: "rf" (Random Forest) or "svm"
CLASSIFIER     = "rf"
# ──────────────────────────────────────────────────────────────────────────────


def load_and_clean(csv_path):
    """Load CSV, drop rows with NaN landmarks, return X and raw labels."""
    if not os.path.exists(csv_path):
        sys.exit(f"❌  Dataset not found: {csv_path}\n"
                 "    Run collect_data.py first.")

    df = pd.read_csv(csv_path)
    print(f"✔  Loaded {len(df)} rows from {csv_path}")

    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"   Dropped {dropped} rows with NaN values (no-hand frames).")

    if df.empty:
        sys.exit("❌  No usable rows after cleaning.")

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return X, y, feature_cols


def build_model(clf_type):
    if clf_type == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale",
                  probability=True, random_state=RANDOM_STATE)
    else:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    # Wrap in a pipeline with scaling (helps SVM; harmless for RF)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


def plot_confusion_matrix(model, X_test, y_test, label_names, out_path):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=label_names)
    fig, ax = plt.subplots(figsize=(max(6, len(label_names)), max(5, len(label_names) - 1)))
    disp = ConfusionMatrixDisplay(cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"📊  Confusion matrix saved  →  {out_path}")


def main():
    print("═" * 55)
    print("   Sign Language – Model Training")
    print("═" * 55)

    # 1. Load data
    X, y_raw, feature_cols = load_and_clean(CSV_PATH)

    # 2. Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes = le.classes_
    print(f"\nClasses ({len(classes)}): {list(classes)}")

    counts = pd.Series(y_raw).value_counts()
    print("\nFrame counts per label:")
    print(counts.to_string())

    # Warn if any class has very few samples
    low = counts[counts < 20]
    if not low.empty:
        print(f"\n⚠  Low sample count for: {list(low.index)}"
              "  — consider collecting more data.")

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # 4. Build & train
    print(f"\nTraining {CLASSIFIER.upper()} …")
    model = build_model(CLASSIFIER)
    model.fit(X_train, y_train)

    # 5. Cross-val on full training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 6. Test-set evaluation
    y_pred = model.predict(X_test)
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    acc = (y_pred == y_test).mean()
    print(f"Test accuracy:      {acc:.3f}")

    report = classification_report(y_test_labels, y_pred_labels)
    print(f"\n{report}")

    with open(REPORT_PATH, "w") as f:
        f.write(f"Test accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"📄  Report saved  →  {REPORT_PATH}")

    # 7. Confusion matrix
    plot_confusion_matrix(model, X_test, y_test, list(range(len(classes))), CM_PATH)

    # 8. Feature importance (RF only)
    if CLASSIFIER == "rf":
        importances = model.named_steps["clf"].feature_importances_
        top_idx     = np.argsort(importances)[-10:][::-1]
        print("\nTop 10 most important landmarks:")
        for i, idx in enumerate(top_idx, 1):
            print(f"  {i:2d}. {feature_cols[idx]:<12s}  importance={importances[idx]:.4f}")

    # 9. Save model & label encoder
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le,    LABEL_MAP_PATH)
    print(f"\n✅  Model saved      →  {MODEL_PATH}")
    print(f"✅  Label map saved  →  {LABEL_MAP_PATH}")
    print("═" * 55)


if __name__ == "__main__":
    main()
