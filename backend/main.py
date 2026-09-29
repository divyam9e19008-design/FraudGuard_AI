from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import numpy as np
import re
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
POLICY_DIR = ROOT / "data"
POLICY_FILES = sorted(POLICY_DIR.glob("*.txt"))
def load_policy_chunks():
    all_chunks = []
    chunk_metadata = []

    for file_path in POLICY_FILES:
        text = file_path.read_text(encoding="utf-8")

        parts = re.split(r"\n(?=\d+\.\s)", text.strip())

        for part in parts:
            part = part.strip()

            if part:
                all_chunks.append(part)
                chunk_metadata.append({
                    "source": file_path.name
                })

    return all_chunks, chunk_metadata
ALL_CHUNKS, CHUNK_METADATA = load_policy_chunks()

BM25_INDEX = BM25Okapi(
    [chunk.lower().split() for chunk in ALL_CHUNKS]
)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

CHUNK_EMBEDDINGS = EMBED_MODEL.encode(
    ALL_CHUNKS,
    convert_to_numpy=True
)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)
def normalize_scores(scores):
    scores = np.array(scores)

    if scores.max() - scores.min() == 0:
        return np.zeros_like(scores)

    return (scores - scores.min()) / (
        scores.max() - scores.min()
    )


def hybrid_search(query, top_k=3, alpha=0.5):
    bm25_scores = BM25_INDEX.get_scores(
        query.lower().split()
    )

    query_embedding = EMBED_MODEL.encode(
        [query],
        convert_to_numpy=True
    )

    # Cosine similarity
    embedding_scores = np.dot(
        CHUNK_EMBEDDINGS,
        query_embedding[0]
    ) / (
        np.linalg.norm(CHUNK_EMBEDDINGS, axis=1)
        * np.linalg.norm(query_embedding[0])
        + 1e-10
    )

    bm25_norm = normalize_scores(bm25_scores)
    embedding_norm = normalize_scores(embedding_scores)

    final_scores = (
        alpha * embedding_norm
        + (1 - alpha) * bm25_norm
    )

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    return[
        {
            "text": ALL_CHUNKS[i],
            "source": CHUNK_METADATA[i]["source"],
            "bm25": float(bm25_norm[i]),
            "embedding": float(embedding_norm[i]),
            "score": float(final_scores[i])
        }
        for i in top_indices
    ]
def rag_answer(query, top_k=3):
    results = hybrid_search(query, top_k)

    context = ""
    for r in results:
        context += f"[Source: {r['source']}]\n{r['text']}\n\n"

    system_prompt = """
You are a fraud investigation assistant.

Answer the user's question using ONLY the policy context provided.

If the question is not answered by the policy context, reply with EXACTLY:
NOT_COVERED

Otherwise:
- Answer clearly and concisely.
- Do not use outside knowledge.
- Always mention the source file(s) used.
"""

    user_prompt = f"""
POLICY CONTEXT:
{context}

QUESTION:
{query}

Answer based only on the above policy context.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content, results

MODEL_PATH = MODEL_DIR / "rf_smote.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

app = FastAPI(title="FraudGuard AI API")

class AskRequest(BaseModel):
    q: str


@app.post("/api/ask")
def ask(request: AskRequest):
    try:
        answer, sources = rag_answer(request.q, top_k=3)

        return {
            "question": request.q,
            "answer": answer,
            "sources": [
                {
                    "source": r["source"],
                    "score": round(r["score"], 4),
                    "text": r["text"]
                }
                for r in sources
            ],
            "model": "BM25 + all-MiniLM-L6-v2 + Groq gpt-oss-120b"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Application(BaseModel):
    income: float
    name_email_similarity: float
    prev_address_months_count: float
    current_address_months_count: float
    customer_age: float
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: float
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: float
    date_of_birth_distinct_emails_4w: float
    employment_status: str
    credit_risk_score: float
    email_is_free: int
    housing_status: str
    phone_home_valid: int
    phone_mobile_valid: int
    bank_months_count: float
    has_other_cards: int
    proposed_credit_limit: float
    foreign_request: int
    source: str
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: int
    device_distinct_emails_8w: float
    month: float

def load_artifacts():
    missing = [str(p.name) for p in [MODEL_PATH, SCALER_PATH, FEATURES_PATH] if not p.exists()]
    if missing:
        raise RuntimeError("Missing model artifacts: " + ", ".join(missing) + ". Run train_model.py first.")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = json.loads(FEATURES_PATH.read_text())
    return model, scaler, features

@app.get("/api/health")
def health():
    try:
        load_artifacts()
        return {"status": "ok", "model": "RandomForest + SMOTE"}
    except RuntimeError as e:
        return {"status": "setup_required", "detail": str(e)}

@app.post("/api/predict")
def predict(application: Application):
    try:
        model, scaler, features = load_artifacts()
        raw = application.model_dump()
        df = pd.DataFrame([raw])

        # Same special missing-value treatment documented in Phase 1.
        low_missing_cols = [
            "current_address_months_count",
            "session_length_in_minutes",
            "credit_risk_score",
            "device_distinct_emails_8w",
        ]
        # For live input, -1 means unknown. The training medians are stored by train_model.py.
        medians = json.loads((MODEL_DIR / "training_medians.json").read_text())
        for col in low_missing_cols:
            if float(df.loc[0, col]) == -1:
                df.loc[0, col] = medians[col]

        categorical_cols = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df = df.reindex(columns=features, fill_value=0)

        X = scaler.transform(df)
        probability = float(model.predict_proba(X)[0, 1])
        score = round(probability * 100, 2)
        category = "LOW" if score < 30 else "MEDIUM" if score < 70 else "HIGH"

        return {
            "fraud_probability": probability,
            "risk_score": score,
            "risk_category": category,
            "model": "RandomForest + SMOTE",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ============================================================
# INVESTIGATE API
# ============================================================

INVESTIGATE_CASES = {
    "TXN-824916": {
        "probability": 0.20,
        "risk_score": 20.0,
        "risk_category": "LOW",
        "rank": "1st of 100",
        "ground_truth": 0
    },

    "TXN-617362": {
        "probability": 0.17,
        "risk_score": 17.0,
        "risk_category": "LOW",
        "rank": "2nd of 100",
        "ground_truth": 0
    },

    "TXN-665979": {
        "probability": 0.09,
        "risk_score": 9.0,
        "risk_category": "LOW",
        "rank": "below top 5",
        "ground_truth": 1
    }
}


@app.get("/api/investigate/{case_id}")
def investigate(case_id: str):
    if case_id not in INVESTIGATE_CASES:
        raise HTTPException(
            status_code=404,
            detail="Case not found"
        )

    return {
        "case_id": case_id,
        **INVESTIGATE_CASES[case_id]
    }
