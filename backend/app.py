"""
app.py — FastAPI server for the 20-newsgroups text classifier.
Start with:
    uvicorn app:app --reload --port 8000
"""

import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

MODEL_DIR = "model"
MODEL_PATH      = os.path.join(MODEL_DIR, "svm_model.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
LABELS_PATH     = os.path.join(MODEL_DIR, "target_names.joblib")
METRICS_PATH    = os.path.join(MODEL_DIR, "metrics.joblib")

# ── Boot-time check ──────────────────────────────────────────────────────────
for path in (MODEL_PATH, VECTORIZER_PATH, LABELS_PATH):
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model artefact not found: {path}\n"
            "Please run  `python train.py`  first."
        )

model        = joblib.load(MODEL_PATH)
vectorizer   = joblib.load(VECTORIZER_PATH)
target_names = list(joblib.load(LABELS_PATH))
metrics = joblib.load(METRICS_PATH) if os.path.exists(METRICS_PATH) else None

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SVM Text Classifier",
    description="20-Newsgroups text classifier powered by LinearSVC + TF-IDF",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class TopPrediction(BaseModel):
    label: str
    label_index: int
    score: float


class PredictResponse(BaseModel):
    label: str
    label_index: int
    top_predictions: list[TopPrediction]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    payload = {"status": "ok", "num_classes": len(target_names)}
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


@app.get("/categories")
def get_categories():
    return {"categories": target_names}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        X = vectorizer.transform([req.text])
        decision_scores = model.decision_function(X)[0]
        normalized_scores = normalize_scores(decision_scores)
        pred_index = int(np.argmax(decision_scores))
        pred_label = target_names[pred_index]

        top_n = 5
        top_indices = np.argsort(normalized_scores)[-top_n:][::-1]
        top_predictions = [
            TopPrediction(
                label=target_names[i],
                label_index=int(i),
                score=round(float(normalized_scores[i]), 4),
            )
            for i in top_indices
        ]

        return PredictResponse(
            label=pred_label,
            label_index=pred_index,
            top_predictions=top_predictions,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
