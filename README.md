# 🧠 SVM Text Classifier — 20 Newsgroups

> **BML Final Project** · Support Vector Machine for multi-class text classification  
> Dataset: [`fetch_20newsgroups`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) (20 topic categories, ~20 000 posts)

---

## Project Structure

```text
.
├── backend/
│   ├── app.py           # FastAPI REST API
│   ├── train.py         # Training script (run once)
│   └── requirements.txt
└── frontend/
    ├── index.html       # Single-page UI
    ├── style.css
    └── app.js
```

---

## Quick Start

### 1 — Backend

```bash
cd backend

# Create & activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (downloads ~14 MB dataset, takes ~1–2 min)
python train.py

# Start the API server
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`  
Auto-generated docs: `http://localhost:8000/docs`

### 2 — Frontend

No build step needed — just open the file:

```bash
# macOS / Linux
open frontend/index.html

# Or serve it with Python (avoids any CORS edge cases)
python3 -m http.server 5500 --directory frontend
# then visit http://localhost:5500
```

The frontend reads its API URL from `frontend/config.js`:

```js
window.APP_CONFIG = {
  apiBaseUrl: "http://localhost:8000"
};
```

For local use, leave it as `http://localhost:8000`.
For GitHub Pages, change it to your deployed backend URL, for example:

```js
window.APP_CONFIG = {
  apiBaseUrl: "https://your-backend-service.onrender.com"
};
```

GitHub Pages can host the frontend only. The FastAPI backend must be deployed separately to a service such as Render, Railway, or Azure.

### 3 — Deploy backend to Render

This repository now includes `render.yaml` so the API can be deployed as a Render Blueprint.

Render will:

1. Install `backend/requirements.txt`
2. Run `python train.py` during build
3. Start FastAPI with Uvicorn

Steps:

1. Push this repository to GitHub.
2. Sign in to Render.
3. Create a new `Blueprint` service from this repository.
4. Render will detect `render.yaml` automatically.
5. After deployment finishes, copy the backend URL.
6. Update `frontend/config.js` with that backend URL.
7. Push again so GitHub Pages uses the live API.

### 4 — Deploy frontend to GitHub Pages

This repository includes a GitHub Actions workflow that publishes the `frontend/` folder to GitHub Pages.

Before enabling it:

```bash
# Edit the frontend config to point to your deployed backend
open frontend/config.js
```

Then on GitHub:

1. Go to your repository settings.
2. Open `Settings > Pages`.
3. Set `Source` to `GitHub Actions`.
4. Push to `main` again if needed.

Your frontend will be published at:

```text
https://NickCT1612yuri.github.io/SVM-text-classifier/
```

---

## API Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/health` | Health check |
| `GET` | `/categories` | List all 20 category names |
| `POST` | `/predict` | Classify text → top-5 predictions |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA discovered a new exoplanet in the habitable zone."}'
```

### Example response

```json
{
  "label": "sci.space",
  "label_index": 14,
  "top_predictions": [
    { "label": "sci.space",        "label_index": 14, "confidence": 0.8231 },
    { "label": "sci.med",          "label_index": 13, "confidence": 0.0412 },
    { "label": "sci.electronics",  "label_index": 12, "confidence": 0.0318 },
    { "label": "sci.crypt",        "label_index": 11, "confidence": 0.0204 },
    { "label": "comp.graphics",    "label_index": 1,  "confidence": 0.0171 }
  ]
}
```

---

## Model Details

| Component | Choice |
| --------- | ------ |
| Vectorizer | TF-IDF (unigrams + bigrams, 80k features, sublinear TF) |
| Classifier | `LinearSVC` wrapped in `CalibratedClassifierCV` (3-fold) |
| Preprocessing | Preserve 20 Newsgroups metadata; sklearn `stop_words="english"` |
| Training split | `subset="train"` with original posts retained |

> Current test-set accuracy for this pipeline: **0.8659** (macro F1 **0.8604**)

---

## Requirements

- Python 3.9+
- See `backend/requirements.txt`

---

## License

For academic use only — BML course final project.
