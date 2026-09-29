# FraudGuard AI — Live ML Integration

## What is now connected
The Detect page contains the real bank-account application inputs from the Phase-1 notebook.

Flow:

Browser form
→ POST /api/predict
→ same notebook-style preprocessing
→ StandardScaler
→ SMOTE-trained Random Forest
→ fraud probability
→ risk score = probability × 100
→ LOW / MEDIUM / HIGH

## One-time setup

From this folder:

```bash
cd backend
pip install -r requirements.txt
```

Put the dataset file here:

```text
data/Base.csv
```

Then:

```bash
python train_model.py
```

This creates the model artifacts in `models/`.

Start API:

```bash
uvicorn main:app --reload --port 8000
```

Then open:

```text
frontend/index.html
```

and go to Detect.

## Important
The backend intentionally does not fabricate a prediction when the model artifacts are missing. The Detect page shows API OFFLINE until the real model is trained and available.

Phase-2 RAG is not silently mocked into this backend yet. It should be integrated after the live Phase-1 prediction path is verified.
