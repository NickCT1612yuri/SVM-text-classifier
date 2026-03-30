"""
train.py — Trains and saves the SVM text-classification model.
Run this once before starting the API server:
    python train.py
"""

import os
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "target_names.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.joblib")


def train_and_save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/4] Loading 20-newsgroups dataset …")
    train_data = fetch_20newsgroups(
        subset="train",
        remove=(),
        shuffle=True,
        random_state=42,
    )
    test_data = fetch_20newsgroups(
        subset="test",
        remove=(),
    )

    X_train = train_data.data
    X_test = test_data.data
    y_train = train_data.target
    y_test = test_data.target
    target_names = train_data.target_names

    # ── Vectorise ────────────────────────────────────────────────────────────
    print("[2/4] Vectorising with TF-IDF …")
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ── Train LinearSVC ──────────────────────────────────────────────────────
    print("[3/4] Training LinearSVC …")
    model = LinearSVC(C=1.0, max_iter=5000)
    model.fit(X_train_vec, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # ── Persist ──────────────────────────────────────────────────────────────
    print("[4/4] Saving model artefacts …")
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
    }
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(target_names, LABELS_PATH)
    joblib.dump(metrics, METRICS_PATH)
    print(f"  ✓ model      → {MODEL_PATH}")
    print(f"  ✓ vectorizer → {VECTORIZER_PATH}")
    print(f"  ✓ labels     → {LABELS_PATH}")
    print(f"  ✓ metrics    → {METRICS_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    train_and_save_model()
