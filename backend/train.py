"""
train.py — Trains and saves the SVM text-classification model.
Run this once before starting the API server:
    python train.py
"""

import os
import re
import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LABELS_PATH = os.path.join(MODEL_DIR, "target_names.joblib")


def preprocess(text: str) -> str:
    """Lightweight text cleaning."""
    text = re.sub(r"\S+@\S+", " ", text)       # remove e-mails
    text = re.sub(r"http\S+|www\S+", " ", text) # remove URLs
    text = re.sub(r"\d+", " ", text)            # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("[1/4] Loading 20-newsgroups dataset …")
    train_data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    test_data = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
    )

    X_train = [preprocess(t) for t in train_data.data]
    X_test  = [preprocess(t) for t in test_data.data]
    y_train = train_data.target
    y_test  = test_data.target
    target_names = train_data.target_names

    # ── Vectorise ────────────────────────────────────────────────────────────
    print("[2/4] Vectorising with TF-IDF …")
    vectorizer = TfidfVectorizer(
        max_features=80_000,
        sublinear_tf=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # ── Train LinearSVC wrapped for probability estimates ────────────────────
    print("[3/4] Training LinearSVC …")
    base_svm = LinearSVC(C=1.0, max_iter=2000, dual=True)
    model = CalibratedClassifierCV(base_svm, cv=3)
    model.fit(X_train_vec, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_vec)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ── Persist ──────────────────────────────────────────────────────────────
    print("[4/4] Saving model artefacts …")
    joblib.dump(model,       MODEL_PATH)
    joblib.dump(vectorizer,  VECTORIZER_PATH)
    joblib.dump(target_names, LABELS_PATH)
    print(f"  ✓ model      → {MODEL_PATH}")
    print(f"  ✓ vectorizer → {VECTORIZER_PATH}")
    print(f"  ✓ labels     → {LABELS_PATH}")
    print("\nTraining complete!")


if __name__ == "__main__":
    train_and_save_model()
